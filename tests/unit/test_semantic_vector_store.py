from app.repositories.semantic_vector_store import SemanticVectorStore
from app.schemas.semantic import SemanticCard
from app.services.semantic_knowledge_service import semantic_card_to_document
from scripts.rebuild_semantic_knowledge_base import rebuild_semantic_knowledge_base


def _card() -> SemanticCard:
    return SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号", "phone"],
        common_field_names=["phone"],
        value_features=["数字字符"],
        description="联系自然人的移动电话号码",
        semantic_category=["联系方式"],
        regulation_keywords=["电话号码"],
    )


def test_semantic_store_returns_card_and_unmodified_raw_score():
    document = semantic_card_to_document(_card())

    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, k):
            assert query == "semantic query"
            assert k == 3
            return [(document, 1.2)]

    result = SemanticVectorStore(client=FakeStore()).search("semantic query", k=3)

    assert result[0].card == _card()
    assert result[0].raw_score == 1.2


def test_semantic_rebuild_loads_before_reset(tmp_path):
    path = tmp_path / "cards.json"
    path.write_text("not json", encoding="utf-8")

    class ForbiddenStore:
        def reset(self):
            raise AssertionError("invalid cards must not reset the store")

    try:
        rebuild_semantic_knowledge_base(path, vector_store=ForbiddenStore())
    except ValueError:
        pass
    else:
        raise AssertionError("invalid JSON must fail")


def test_semantic_rebuild_replaces_only_injected_store(tmp_path):
    import json

    path = tmp_path / "cards.json"
    path.write_text(json.dumps([_card().model_dump()]), encoding="utf-8")

    class CapturingStore:
        def __init__(self):
            self.reset_calls = 0
            self.documents = []

        def reset(self):
            self.reset_calls += 1

        def add_documents(self, documents):
            self.documents = documents

    store = CapturingStore()
    count = rebuild_semantic_knowledge_base(path, vector_store=store)

    assert count == 1
    assert store.reset_calls == 1
    assert store.documents[0].metadata["semantic_type"] == "手机号码"

