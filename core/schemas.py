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
    sample_values: List[str] = Field(default_factory=list)
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


class ClassificationOutput(BaseModel):
    """LLM 结构化输出 — 仅包含分类判定本身，不包含系统侧附加信息。"""

    category: str = Field(description="数据分类，如: 个人信息、敏感个人信息、财务数据、员工数据、业务经营数据")
    subcategory: Optional[str] = Field(default=None, description="细分类，如: 个人联系方式、身份标识信息、薪酬信息")
    level: Literal["L1", "L2", "L3", "L4"] = Field(description="敏感等级，L1=公开/L2=内部/L3=敏感/L4=核心敏感")
    confidence: float = Field(description="置信度 0.0-1.0，依据不足时不超过 0.75")
    reason: str = Field(description="简要说明判断理由")
    need_review: bool = Field(description="依据不足或字段含义模糊时设为 true")


class ClassificationResult(BaseModel):
    """系统侧完整分类结果 — 在 LLM 输出之上附加检索依据和决策路径。"""

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
