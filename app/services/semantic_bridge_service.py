"""Experiment E two-stage Semantic Bridge classification orchestration."""

import logging

from app.rag.prompt import build_classification_prompt
from app.rag.semantic_bridge import (
    build_objective_profile,
    build_regulation_bridge_query,
    build_semantic_query,
)
from app.schemas.classification import ClassificationResult, SemanticBridgeTrace
from app.schemas.field import FieldProfile
from app.services.value_profiler import ValueProfiler

logger = logging.getLogger(__name__)


class SemanticBridgeClassificationService:
    """Profile objectively, retrieve semantics, retrieve laws, then call one LLM."""

    def __init__(
        self,
        semantic_vector_store,
        regulation_vector_store,
        llm_service,
        *,
        value_profiler=None,
        semantic_top_k: int = 3,
        regulation_top_k: int = 3,
    ) -> None:
        self.semantic_vector_store = semantic_vector_store
        self.regulation_vector_store = regulation_vector_store
        self.llm_service = llm_service
        self.value_profiler = value_profiler or ValueProfiler()
        self.semantic_top_k = semantic_top_k
        self.regulation_top_k = regulation_top_k

    def classify_field(self, field: FieldProfile | dict) -> ClassificationResult:
        field_profile = FieldProfile.model_validate(field)
        trace = SemanticBridgeTrace()
        stage = "profiling"
        try:
            trace.profiling = build_objective_profile(
                field_profile,
                self.value_profiler,
            )
            trace.semantic_query = build_semantic_query(
                field_profile,
                trace.profiling,
            )
            stage = "semantic_retrieval"
            trace.semantic_retrieval = self.semantic_vector_store.search(
                trace.semantic_query,
                k=self.semantic_top_k,
            )
            if not trace.semantic_retrieval:
                raise RuntimeError("Semantic KB 未检索到可用语义卡片")

            selected = trace.semantic_retrieval[0]
            trace.selected_card = selected.card
            trace.selected_semantic_type = selected.card.semantic_type
            if len(trace.semantic_retrieval) >= 2:
                trace.top1_top2_score_gap = round(
                    selected.raw_score - trace.semantic_retrieval[1].raw_score,
                    12,
                )
            trace.regulation_query = build_regulation_bridge_query(
                field_profile,
                selected.card,
            )
            stage = "regulation_retrieval"
            trace.regulation_retrieval = self.regulation_vector_store.search_raw(
                trace.regulation_query,
                k=self.regulation_top_k,
            )
            if not trace.regulation_retrieval:
                raise RuntimeError("法规知识库未检索到可用依据")

            evidence = [item.evidence for item in trace.regulation_retrieval]
            stage = "llm_classification"
            output = self.llm_service.classify(
                build_classification_prompt(
                    field_profile,
                    evidence,
                    value_profile=trace.profiling,
                    semantic_knowledge={
                        "card": selected.card.model_dump(),
                        "raw_score": selected.raw_score,
                    },
                )
            )
            return ClassificationResult(
                field_name=field_profile.field_name,
                is_personal=output.is_personal,
                category=output.category,
                subcategory=output.subcategory,
                level=output.level,
                confidence=output.confidence,
                reason=output.reason,
                evidence=evidence,
                need_review=output.need_review,
                decision_path="semantic_rag_llm",
                experiment_trace=trace,
            )
        except Exception:
            trace.failed_stage = stage
            logger.exception(
                "Experiment E 字段 %s 在 %s 阶段失败",
                field_profile.field_name,
                stage,
            )
            return ClassificationResult(
                field_name=field_profile.field_name,
                is_personal=None,
                category="未知",
                level="UNKNOWN",
                confidence=0.0,
                reason="分类处理失败，请进行人工复核。",
                evidence=[item.evidence for item in trace.regulation_retrieval],
                need_review=True,
                decision_path="semantic_rag_llm_error",
                experiment_trace=trace,
            )
