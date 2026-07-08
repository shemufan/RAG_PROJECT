# backend/services/rag_classifier.py
"""RAG 分类器 — 核心分类管线（Week3 占位，后续实现具体逻辑）。"""

import logging

logger = logging.getLogger(__name__)


class RAGClassifier:
    """RAG 分类管线：错题本缓存 → 向量检索 → LLM 推理。

    当前为 Week3 占位实现，后续将集成 ChromaStore 和 LLMService。
    """

    def __init__(self, chroma_store=None, llm_service=None, error_cache=None):
        self.chroma_store = chroma_store
        self.llm_service = llm_service
        self.error_cache = error_cache if error_cache is not None else {}

    def classify_field(self, field_info: dict) -> dict:
        """对单个字段执行分类定级（Week3 mock 实现）。

        Args:
            field_info: 字段信息字典，包含 field_name, field_comment,
                        table_name, sample_value。

        Returns:
            dict: 分类结果，包含 level, reason, matched_rules, references,
                  confidence, need_manual_review。
        """
        logger.info("classify_field mock 调用: %s", field_info.get("field_name"))

        field_name = field_info.get("field_name", "")
        sample_value = field_info.get("sample_value", "")

        # 基于样例值做简单 mock 判断
        if sample_value and any(keyword in str(sample_value) for keyword in ["@", ".com", "http"]):
            # 样例值包含邮箱或网址特征 → 个人信息
            return {
                "level": "L2",
                "reason": "样例值包含邮箱/网址特征，可能属于个人联系方式。",
                "matched_rules": ["个人信息保护法 第4条"],
                "references": ["《中华人民共和国个人信息保护法》"],
                "confidence": 0.70,
                "need_manual_review": False,
            }

        if sample_value and any(keyword in str(sample_value) for keyword in ["138", "139", "1", "万", "元", "￥"]):
            # 样例值包含数值特征 → 可能敏感
            return {
                "level": "L3",
                "reason": "样例值包含数值信息，可能涉及财务或交易数据。",
                "matched_rules": ["个人信息保护法 第4条", "数据安全法 第21条"],
                "references": ["《中华人民共和国个人信息保护法》", "《中华人民共和国数据安全法》"],
                "confidence": 0.65,
                "need_manual_review": False,
            }

        # 默认 mock 结果 — 无样例值时返回 L1
        return {
            "level": "L1",
            "reason": f"Mock 判定：字段 '{field_name}' 无明显敏感特征，暂定为公开。",
            "matched_rules": ["数据安全法 第21条"],
            "references": ["《中华人民共和国数据安全法》"],
            "confidence": 0.60,
            "need_manual_review": True,
        }

    def set_error_log_cache(self, cache: dict):
        """更新错题本缓存。"""
        self.error_cache.clear()
        self.error_cache.update(cache)
