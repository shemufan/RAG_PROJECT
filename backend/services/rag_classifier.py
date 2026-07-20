"""Retrieval-augmented field classification for the Week3 baseline."""

import json
import logging

from backend.schemas.classify_schema import (
    ClassificationResult,
    Evidence,
    FieldProfile,
)
from backend.services.prompt import CLASSIFICATION_USER

logger = logging.getLogger(__name__)


class RAGClassifier:
    """Run vector retrieval followed by structured LLM classification."""

    def __init__(self, chroma_store=None, llm_service=None, error_cache=None):
        self.chroma_store = chroma_store
        self.llm_service = llm_service
        self.error_cache = error_cache if error_cache is not None else {}

    def build_query_text(self, field: FieldProfile) -> str:
        labels = {
            "field_name": field.field_name,
            "field_cn": field.field_cn,
            "field_comment": field.field_comment,
            "data_type": field.data_type,
            "sample_values": "、".join(field.sample_values),
            "business_domain": field.business_domain,
            "table_name": field.table_name,
            "database_name": field.database_name,
        }
        return "\n".join(f"{key}: {value}" for key, value in labels.items() if value)

    def retrieve_evidence(self, field: FieldProfile, k: int = 3) -> list[Evidence]:
        if self.chroma_store is None:
            raise RuntimeError("ChromaStore 未初始化")

        query = self.build_query_text(field)
        try:
            if hasattr(self.chroma_store, "similarity_search_with_relevance_scores"):
                rows = self.chroma_store.similarity_search_with_relevance_scores(query, k=k)
            else:
                rows = [(doc, None) for doc in self.chroma_store.similarity_search(query, k=k)]
        except Exception as exc:
            raise RuntimeError(f"知识库检索失败: {exc}") from exc

        evidence = []
        for document, score in rows:
            metadata = document.metadata or {}
            evidence.append(
                Evidence(
                    content=document.page_content,
                    source=metadata.get("document_name") or metadata.get("source") or "未知来源",
                    article=metadata.get("article") or metadata.get("hierarchy_level"),
                    score=max(0.0, min(1.0, float(score))) if score is not None else None,
                )
            )
        return evidence

    def classify_field(self, field: FieldProfile | dict) -> ClassificationResult:
        if isinstance(field, dict):
            field = FieldProfile.model_validate(field)

        try:
            evidence = self.retrieve_evidence(field, k=3)
            if not evidence:
                raise RuntimeError("知识库未检索到可用依据")
            if self.llm_service is None:
                raise RuntimeError("LLMService 未初始化")

            user_message = CLASSIFICATION_USER.format(
                field_profile=json.dumps(field.model_dump(), ensure_ascii=False, indent=2),
                evidence=json.dumps(
                    [item.model_dump() for item in evidence], ensure_ascii=False, indent=2
                ),
            )
            output = self.llm_service.classify(user_message)
            return ClassificationResult(
                field_name=field.field_name,
                category=output.category,
                subcategory=output.subcategory,
                level=output.level,
                confidence=output.confidence,
                reason=output.reason,
                evidence=evidence,
                need_review=output.need_review,
                decision_path="rag_llm",
            )
        except Exception as exc:
            logger.exception("字段 %s 分类失败", field.field_name)
            return ClassificationResult(
                field_name=field.field_name,
                category="未知",
                level="UNKNOWN",
                confidence=0.0,
                reason=f"分类失败，建议人工复核：{exc}",
                evidence=locals().get("evidence", []),
                need_review=True,
                decision_path="rag_llm_error",
            )

    def set_error_log_cache(self, cache: dict):
        """Retained for compatibility with the post-Week3 batch workflow."""
        self.error_cache.clear()
        self.error_cache.update(cache)
