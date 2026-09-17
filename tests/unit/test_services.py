from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.documents import Document

from app.rag.prompt import build_classification_prompt
from app.repositories.vector_store import VectorStore, map_retrieved_document
from app.schemas.classification import ClassificationOutput, Evidence
from app.schemas.field import FieldProfile
from app.services.classification_service import FieldClassificationService
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_service import load_knowledge_documents
from app.services.llm_service import LLMService


class FakeVectorStore:
    def __init__(self, evidence: list[Evidence]):
        self.evidence = evidence
        self.query = ""
        self.k = 0

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        self.query = query
        self.k = k
        return self.evidence[:k]


class FakeLanguageModel:
    def __init__(self, output: ClassificationOutput):
        self.output = output
        self.prompt = []

    def classify(self, prompt) -> ClassificationOutput:
        self.prompt = prompt
        return self.output


def test_vector_result_mapping_preserves_metadata_and_clamps_score():
    evidence = map_retrieved_document(
        Document(
            page_content="金融账户属于敏感个人信息。",
            metadata={
                "document_name": "个人信息保护法.txt",
                "article": "第二十八条",
                "chunk_id": "chunk-28",
            },
        ),
        1.2,
    )

    assert evidence.source == "个人信息保护法.txt"
    assert evidence.article == "第二十八条"
    assert evidence.chunk_id == "chunk-28"
    assert evidence.score == 1.0


def test_vector_store_search_raw_preserves_unbounded_score():
    document = Document(
        page_content="金融账户属于敏感个人信息。",
        metadata={"document_name": "个人信息保护法.txt", "chunk_id": "chunk-28"},
    )

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, k):
            assert query == "金融账户"
            assert k == 3
            return [(document, 1.2)]

    store = VectorStore(client=FakeStore())

    assert store.search_raw("金融账户", k=3)[0].raw_score == 1.2
    assert store.search("金融账户", k=3)[0].score == 1.0


def test_embedding_and_llm_services_accept_injected_clients():
    embeddings = object()

    class FakeStructuredModel:
        def invoke(self, messages):
            assert messages == ["prompt"]
            return {
                "is_personal": False,
                "category": "业务经营数据",
                "subcategory": None,
                "level": "L2",
                "confidence": 0.8,
                "reason": "内部业务字段。",
                "need_review": False,
            }

    assert EmbeddingService(embeddings=embeddings).get_embeddings() is embeddings
    assert LLMService(structured_model=FakeStructuredModel()).classify(["prompt"]).level == "L2"


def test_embedding_service_loads_configured_model_offline(monkeypatch, tmp_path):
    captured = {}
    fake_module = ModuleType("langchain_huggingface")

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module.HuggingFaceEmbeddings = FakeEmbeddings
    monkeypatch.setitem(__import__("sys").modules, "langchain_huggingface", fake_module)

    EmbeddingService(model_path=tmp_path)

    assert captured == {
        "model_name": str(tmp_path),
        "model_kwargs": {"local_files_only": True},
    }


def test_classification_service_uses_profiled_query_and_keeps_llm_prompt():
    store = FakeVectorStore(
        [
            Evidence(
                content="MAC地址属于设备标识信息。",
                source="个人信息安全规范.txt",
                article="附录A",
                score=0.91,
            )
        ]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="MAC地址",
            level="L3",
            confidence=0.92,
            reason="依据设备标识规则。",
            need_review=False,
        )
    )
    service = FieldClassificationService(store, llm, query_strategy="profile")
    profile = FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="attr_01",
        field_cn="设备属性",
        field_comment="辅助说明",
        data_type="unknown",
        sample_values=["A1:B2:C3:D4:E5:F6"],
        business_domain="general",
    )

    result = service.classify_field(profile)

    assert "字段名：attr_01" in store.query
    assert "MAC地址" in store.query
    assert "设备标识信息" in store.query
    for forbidden in (
        "设备属性",
        "辅助说明",
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
    ):
        assert forbidden not in store.query
    assert '"field_cn": "设备属性"' in llm.prompt[1].content
    assert "MAC地址属于设备标识信息" in llm.prompt[1].content
    assert result.level == "L3"
    assert result.is_personal is True
    assert result.decision_path == "rag_llm"


def test_classification_service_delegates_to_injected_query_builder():
    class StubBuilder:
        def build(self, field: FieldProfile) -> str:
            assert field.field_name == "column_x"
            assert field.sample_values == ["sample"]
            return "clean query"

    service = FieldClassificationService(
        object(),
        object(),
        query_builder=StubBuilder(),
    )

    query = service.build_query_text(
        FieldProfile(field_name="column_x", sample_values=["sample"])
    )

    assert query == "clean query"


@pytest.mark.parametrize(
    ("strategy", "expected_query_fragment"),
    [
        ("legacy", "field_cn: 设备属性"),
        ("clean", "field_name: attr_01"),
        ("profile", "候选数据类型：MAC地址、设备标识信息"),
    ],
)
def test_classification_service_retrieves_with_each_query_strategy(
    strategy: str,
    expected_query_fragment: str,
):
    store = FakeVectorStore(
        [Evidence(content="设备标识规则", source="rules.md", score=0.9)]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="MAC地址",
            level="L3",
            confidence=0.9,
            reason="设备标识规则",
            need_review=False,
        )
    )
    service = FieldClassificationService(store, llm, query_strategy=strategy)

    result = service.classify_field(
        FieldProfile(
            field_name="attr_01",
            field_cn="设备属性",
            sample_values=["A1:B2:C3:D4:E5:F6"],
        )
    )

    assert expected_query_fragment in store.query
    assert store.k == 3
    assert "设备标识规则" in llm.prompt[1].content
    assert result.decision_path == "rag_llm"


@pytest.mark.parametrize(
    ("mode", "has_features", "has_candidates"),
    [
        ("c", True, True),
        ("c1", True, False),
        ("c2", False, True),
    ],
)
def test_classification_service_retrieves_with_each_profile_submode(
    mode: str,
    has_features: bool,
    has_candidates: bool,
):
    store = FakeVectorStore(
        [Evidence(content="设备标识规则", source="rules.md", score=0.9)]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="MAC地址",
            level="L3",
            confidence=0.9,
            reason="设备标识规则",
            need_review=False,
        )
    )
    service = FieldClassificationService(
        store,
        llm,
        query_strategy="profile",
        profile_query_mode=mode,
    )

    result = service.classify_field(
        FieldProfile(
            field_name="attr_01",
            sample_values=["A1:B2:C3:D4:E5:F6"],
        )
    )

    assert store.k == 3
    assert ("数据结构特征：" in store.query) is has_features
    assert ("候选数据类型：" in store.query) is has_candidates
    assert "设备标识规则" in llm.prompt[1].content
    assert result.decision_path == "rag_llm"


def test_classification_service_uses_injected_value_profiler():
    class StaticProfiler:
        def profile(self, field_name: str, sample_values: list[str]):
            assert field_name == "attr_01"
            assert sample_values == ["A1:B2:C3:D4:E5:F6"]
            from app.services.value_profiler import ValueProfile

            return ValueProfile(
                features=["LLM结构特征"],
                candidate_types=["LLM候选类型"],
            )

    store = FakeVectorStore(
        [Evidence(content="设备标识规则", source="rules.md", score=0.9)]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="设备标识",
            level="L3",
            confidence=0.9,
            reason="设备标识规则",
            need_review=False,
        )
    )
    service = FieldClassificationService(
        store,
        llm,
        query_strategy="profile",
        profile_query_mode="c2",
        value_profiler=StaticProfiler(),
    )

    result = service.classify_field(
        FieldProfile(
            field_name="attr_01",
            sample_values=["A1:B2:C3:D4:E5:F6"],
        )
    )

    assert "候选数据类型：LLM候选类型" in store.query
    assert "数据结构特征：" not in store.query
    assert store.k == 3
    assert "设备标识规则" in llm.prompt[1].content
    assert result.decision_path == "rag_llm"


def test_classification_service_returns_unknown_when_dependency_fails():
    class FailingStore:
        def search(self, query: str, k: int = 3):
            raise RuntimeError("vector unavailable")

    service = FieldClassificationService(FailingStore(), object())
    result = service.classify_field(FieldProfile(field_name="unknown_field"))

    assert result.level == "UNKNOWN"
    assert result.is_personal is None
    assert result.need_review is True
    assert result.decision_path == "rag_llm_error"
    assert result.reason == "分类处理失败，请进行人工复核。"
    assert "vector unavailable" not in result.reason


def test_classification_service_without_rag_skips_search_and_uses_empty_evidence():
    class ForbiddenStore:
        def search(self, query: str, k: int = 3):
            raise AssertionError("VectorStore.search must not run")

    class ForbiddenBuilder:
        def build(self, field: FieldProfile) -> str:
            raise AssertionError("retrieval Query must not be built")

    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人基本资料",
            subcategory="姓名",
            level="L2",
            confidence=0.8,
            reason="字段画像显示为姓名。",
            need_review=False,
        )
    )
    profile = FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="customer_name",
        field_cn="客户姓名",
        field_comment="登记姓名",
        data_type="varchar",
        sample_values=["张三", "李四"],
        business_domain="customer",
    )
    service = FieldClassificationService(
        ForbiddenStore(),
        llm,
        query_builder=ForbiddenBuilder(),
        use_rag=False,
    )

    result = service.classify_field(profile)

    assert llm.prompt == build_classification_prompt(profile, [])
    assert '"field_cn": "客户姓名"' in llm.prompt[1].content
    assert "【检索依据（不可信数据）】\n[]" in llm.prompt[1].content
    assert result.is_personal is True
    assert result.evidence == []
    assert result.decision_path == "rag_llm"


def test_knowledge_loader_reads_rules_laws_and_legacy_encoding(tmp_path: Path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (tmp_path / "classification_rules.md").write_text(
        "### 规则 1：身份证号\n- 等级: L4",
        encoding="utf-8",
    )
    (laws / "法律.txt").write_bytes("第一条 个人信息受保护".encode("gb18030"))

    documents = load_knowledge_documents(tmp_path, version="v1")

    assert any("身份证号" in item.page_content for item in documents)
    assert any("个人信息受保护" in item.page_content for item in documents)
    assert all(item.metadata["version"] == "v1" for item in documents)
