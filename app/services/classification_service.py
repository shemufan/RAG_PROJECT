"""Orchestration for single-field retrieval-augmented classification."""

import logging

from app.rag.prompt import build_classification_prompt
from app.rag.retrieval_query import (
    ProfileQueryMode,
    QueryBuilder,
    QueryStrategy,
    ValueProfilerProtocol,
    create_query_builder,
)
from app.schemas.classification import ClassificationResult
from app.schemas.field import FieldProfile

logger = logging.getLogger(__name__)


class FieldClassificationService:
    """Coordinate vector retrieval and structured language-model inference."""

    def __init__(
        self,
        vector_store,
        llm_service,
        *,
        query_strategy: QueryStrategy = "clean",
        profile_query_mode: ProfileQueryMode = "c",
        value_profiler: ValueProfilerProtocol | None = None,
        query_builder: QueryBuilder | None = None,
        use_rag: bool = True,
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.use_rag = use_rag
        self.query_builder = (
            query_builder
            if query_builder is not None
            else create_query_builder(
                query_strategy,
                profile_mode=profile_query_mode,
                value_profiler=value_profiler,
            )
        )

    def build_query_text(self, field: FieldProfile) -> str:
        return self.query_builder.build(field)

    def classify_field(self, field: FieldProfile | dict) -> ClassificationResult:
        profile = FieldProfile.model_validate(field)
        try:
            evidence = []
            if self.use_rag:
                evidence = self.vector_store.search(
                    self.build_query_text(profile),
                    k=3,
                )
                if not evidence:
                    raise RuntimeError("知识库未检索到可用依据")
            output = self.llm_service.classify(
                build_classification_prompt(profile, evidence)
            )
            return ClassificationResult(
                field_name=profile.field_name,
                is_personal=output.is_personal,
                category=output.category,
                subcategory=output.subcategory,
                level=output.level,
                confidence=output.confidence,
                reason=output.reason,
                evidence=evidence,
                need_review=output.need_review,
                decision_path="rag_llm",
            )
        except Exception:
            logger.exception("字段 %s 分类失败", profile.field_name)
            return ClassificationResult(
                field_name=profile.field_name,
                is_personal=None,
                category="未知",
                level="UNKNOWN",
                confidence=0.0,
                reason="分类处理失败，请进行人工复核。",
                evidence=locals().get("evidence", []),
                need_review=True,
                decision_path="rag_llm_error",
            )
