"""HTTP adapter for single-field classification."""

import logging

from fastapi import APIRouter, Depends, Request

from app.schemas.classification import (
    ClassificationResult,
    ClassifyResponse,
)
from app.schemas.field import FieldProfile
from app.services.classification_service import FieldClassificationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["classification"])


def get_classification_service(request: Request) -> FieldClassificationService:
    return request.app.state.classification_service


@router.post("/classify", response_model=ClassifyResponse)
def classify_field(
    field: FieldProfile,
    service: FieldClassificationService = Depends(get_classification_service),
) -> ClassifyResponse:
    try:
        result = service.classify_field(field)
    except Exception:
        logger.exception("分类接口异常")
        result = ClassificationResult(
            field_name=field.field_name,
            category="未知",
            level="UNKNOWN",
            confidence=0.0,
            reason="系统处理失败，请进行人工复核。",
            evidence=[],
            need_review=True,
            decision_path="api_error",
        )
    if result.level == "UNKNOWN":
        return ClassifyResponse(code=500, message="classify failed", data=result)
    return ClassifyResponse(data=result)
