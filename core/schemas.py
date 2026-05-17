# core/schemas.py

# 定义结构化输出类型
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class FieldProfile(BaseModel):
    database_name: Optional[str] = None
    table_name: Optional[str] = None
    field_name: str
    field_comment: Optional[str] = None
    field_cn: Optional[str] = None
    data_type: Optional[str] = None
    sample_values: List[str] = []
    business_domain: str = "general"
    source_system: Optional[str] = None


class Evidence(BaseModel):
    document_name: str
    source_type: str = "unknown"
    domain: str = "general"
    hierarchy_level: Optional[str] = None
    content: str
    score: Optional[float] = None
    chunk_id: Optional[str] = None


class ClassificationResult(BaseModel):
    field_name: str
    category: str
    subcategory: Optional[str] = None
    level: Literal["L1", "L2", "L3", "L4", "未知"]
    confidence: float = 0.0
    reason: str
    evidence: List[Evidence] = Field(default_factory=list)
    need_review: bool
    decision_path: str = "rag_llm"
    raw_response: Optional[str] = None
