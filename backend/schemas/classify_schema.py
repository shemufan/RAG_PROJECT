# backend/schemas/classify_schema.py
"""分类相关 Pydantic 数据模型 — Week3 接口定义。

只定义请求/响应的数据结构，不包含业务逻辑。
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── 请求模型 ──────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    """单字段分类 API 请求体。

    Attributes:
        field_name:    字段名（必填）。
        field_comment: 字段注释/业务说明。
        table_name:    所属表名。
        sample_value:  样例值，帮助 LLM 理解字段语义。
    """

    field_name: str
    field_comment: Optional[str] = None
    table_name: Optional[str] = None
    sample_value: Optional[str] = None


# ── 响应模型 ──────────────────────────────────────────────

class ClassifyResult(BaseModel):
    """单字段分类结果 — API 响应中 data 字段的内容。

    Attributes:
        level:              敏感等级 (L1/L2/L3/L4/未知)。
        reason:             判定理由简述。
        matched_rules:      命中的法规条款列表。
        references:         引用的法规出处列表。
        confidence:         置信度 0.0-1.0。
        need_manual_review: 是否需要人工复核。
    """

    level: str = ""
    reason: str = ""
    matched_rules: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    need_manual_review: bool = False


class ClassifyResponse(BaseModel):
    """分类 API 统一响应格式。

    Attributes:
        code:    业务状态码，200 表示成功。
        message: 提示信息。
        data:    分类结果。
    """

    code: int = 200
    message: str = "success"
    data: Optional[ClassifyResult] = None
