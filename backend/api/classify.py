# backend/api/classify.py
"""分类相关 API 路由 — 单字段分类接口（Week3）。"""

import logging
from fastapi import APIRouter
from backend.schemas.classify_schema import (
    ClassifyRequest,
    ClassifyResult,
    ClassifyResponse,
)
from backend.services.rag_classifier import RAGClassifier

logger = logging.getLogger(__name__)

router = APIRouter(tags=["classification"])

# 无依赖实例（后续改为 FastAPI Depends 注入）
_classifier = RAGClassifier()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_single(request: ClassifyRequest):
    """对单个字段执行 RAG+LLM 分类定级。

    请求体示例:
        {
            "field_name": "user_phone",
            "field_comment": "用户手机号",
            "table_name": "t_user",
            "sample_value": "13800138000"
        }

    成功响应 data 中包含 ClassifyResult，
    失败时返回 code=500 兜底结果。
    """
    logger.info("收到分类请求: %s", request.field_name)

    try:
        field_info = request.model_dump()
        result_dict = _classifier.classify_field(field_info)
        result = ClassifyResult(**result_dict)
        return ClassifyResponse(code=200, message="success", data=result)

    except Exception as e:
        logger.error("分类失败: %s", str(e))
        fallback = ClassifyResult(
            level="UNKNOWN",
            reason="系统处理失败，建议人工复核。",
            matched_rules=[],
            references=[],
            confidence=0.0,
            need_manual_review=True,
        )
        return ClassifyResponse(
            code=500,
            message=f"classify failed: {str(e)}",
            data=fallback,
        )
