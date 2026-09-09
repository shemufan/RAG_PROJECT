from app.schemas.classification import (
    ClassificationOutput,
    Evidence,
    RegulationRetrievalResult,
)
from app.schemas.field import FieldProfile
from app.schemas.semantic import SemanticCard, SemanticRetrievalResult
from app.services.semantic_bridge_service import SemanticBridgeClassificationService


def _card(name: str, score: float) -> SemanticRetrievalResult:
    return SemanticRetrievalResult(
        card=SemanticCard(
            semantic_type=name,
            aliases=[name],
            common_field_names=["contact_value"],
            value_features=["数字字符"],
            description=f"{name}的通用业务含义",
            semantic_category=["联系方式"],
            regulation_keywords=[name, "个人信息"],
        ),
        raw_score=score,
    )


class ObjectiveProfiler:
    def profile(self, *_args):
        raise AssertionError("semantic detector must not be called")

    def basic_statistics(self, values):
        return ["字符串长度约11位", "数字字符占比100%"]


def test_service_runs_two_retrievals_and_calls_llm_once():
    events = []
    semantic_results = [
        _card("手机号码", 0.82),
        _card("用户ID", 0.61),
        _card("银行卡号", 0.49),
    ]

    class SemanticStore:
        def search(self, query, k):
            events.append("semantic")
            assert "13812345678" in query
            assert k == 3
            return semantic_results

    class RegulationStore:
        def search_raw(self, query, k):
            events.append("regulation")
            assert "字段语义类型：手机号码" in query
            assert "13812345678" not in query
            assert k == 3
            return [
                RegulationRetrievalResult(
                    evidence=Evidence(
                        content="联系方式属于个人信息。",
                        source="rules.md",
                        chunk_id="chunk-1",
                        score=0.71,
                    ),
                    raw_score=0.71,
                )
            ]

    class LLM:
        def __init__(self):
            self.calls = 0

        def classify(self, prompt):
            events.append("llm")
            self.calls += 1
            assert "Semantic Knowledge" in prompt[1].content
            return ClassificationOutput(
                is_personal=True,
                category="个人基本资料",
                subcategory="手机号码",
                level="L3",
                confidence=0.91,
                reason="法规依据与字段语义一致。",
                need_review=False,
            )

    llm = LLM()
    result = SemanticBridgeClassificationService(
        SemanticStore(),
        RegulationStore(),
        llm,
        value_profiler=ObjectiveProfiler(),
    ).classify_field(
        FieldProfile(
            field_name="contact_value",
            field_cn="联系值",
            sample_values=["13812345678", "15987654321"],
        )
    )

    assert events == ["semantic", "regulation", "llm"]
    assert llm.calls == 1
    assert result.decision_path == "semantic_rag_llm"
    assert result.experiment_trace.selected_semantic_type == "手机号码"
    assert result.experiment_trace.top1_top2_score_gap == 0.21
    assert result.experiment_trace.semantic_retrieval == semantic_results
    assert result.experiment_trace.regulation_retrieval[0].raw_score == 0.71


def test_service_preserves_trace_and_skips_llm_when_semantic_retrieval_is_empty():
    class EmptySemanticStore:
        def search(self, query, k):
            return []

    class ForbiddenDependency:
        def __getattr__(self, name):
            raise AssertionError(f"{name} must not be called")

    result = SemanticBridgeClassificationService(
        EmptySemanticStore(),
        ForbiddenDependency(),
        ForbiddenDependency(),
        value_profiler=ObjectiveProfiler(),
    ).classify_field(
        FieldProfile(field_name="unknown", sample_values=["123"])
    )

    assert result.level == "UNKNOWN"
    assert result.decision_path == "semantic_rag_llm_error"
    assert result.experiment_trace.failed_stage == "semantic_retrieval"
    assert result.experiment_trace.semantic_query
    assert result.experiment_trace.semantic_retrieval == []


def test_service_stops_before_llm_when_regulation_retrieval_is_empty():
    class SemanticStore:
        def search(self, query, k):
            return [_card("手机号码", 0.8)]

    class EmptyRegulationStore:
        def search_raw(self, query, k):
            return []

    class ForbiddenLLM:
        def classify(self, prompt):
            raise AssertionError("LLM must not be called without regulation evidence")

    result = SemanticBridgeClassificationService(
        SemanticStore(),
        EmptyRegulationStore(),
        ForbiddenLLM(),
        value_profiler=ObjectiveProfiler(),
    ).classify_field(FieldProfile(field_name="contact", sample_values=["123"]))

    assert result.level == "UNKNOWN"
    assert result.experiment_trace.failed_stage == "regulation_retrieval"
    assert result.experiment_trace.selected_semantic_type == "手机号码"
    assert result.experiment_trace.regulation_query

