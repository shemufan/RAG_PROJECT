"""Single-field classification API for the Week3 baseline."""

import logging

from fastapi import APIRouter, Depends

from backend.api.deps import get_rag_classifier
from backend.schemas.classify_schema import (
    ClassificationResult,
    ClassifyResponse,
    FieldProfile,
)
from backend.services.rag_classifier import RAGClassifier

logger = logging.getLogger(__name__)
router = APIRouter(tags=["classification"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify_single(
    request: FieldProfile,
    classifier: RAGClassifier = Depends(get_rag_classifier),
) -> ClassifyResponse:
    logger.info("收到分类请求: %s", request.field_name)
    try:
        result = classifier.classify_field(request)
        if result.level == "UNKNOWN":
            return ClassifyResponse(code=500, message="classify failed", data=result)
        return ClassifyResponse(code=200, message="success", data=result)
    except Exception as exc:
        logger.exception("分类接口异常")
        fallback = ClassificationResult(
            field_name=request.field_name,
            category="未知",
            level="UNKNOWN",
            confidence=0.0,
            reason=f"系统处理失败，建议人工复核：{exc}",
            evidence=[],
            need_review=True,
            decision_path="api_error",
        )
        return ClassifyResponse(code=500, message=f"classify failed: {exc}", data=fallback)
