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

    def build_query_text(self, field_info: dict) -> str:
        """将字段信息字典拼接为检索查询文本。"""
        parts = []
        for key in ("field_name", "field_comment", "table_name"):
            val = field_info.get(key)
            if val:
                parts.append(str(val))
        return " | ".join(parts) if parts else str(field_info)

    def retrieve_evidence(self, field_info: dict, k) -> list[dict]:
        """从向量库检索相似证据，返回与下游兼容的 dict 列表。

        Args:
            field_info: 字段信息字典
            k: 返回 top-k 相似结果

        Returns:
            list[dict]: 每条证据为一个字典，供 augment 阶段序列化使用
        """
        if self.chroma_store is None:
            logger.warning("ChromaStore 未初始化，跳过向量检索")
            return []

        query_text = self.build_query_text(field_info)

        try:
            docs = self.chroma_store.similarity_search(query_text, k=k)
            # 将 ChromaDB Document 转换为 dict，使下游 .model_dump() / json.dumps 可用
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        except Exception as e:
            logger.error("向量检索失败: %s", str(e))
            return []

    def classify_field(self, field_info: dict) -> dict:
        """对单个字段执行分类定级（Week3 mock 实现）。

        Args:
            field_info: 字段信息字典，包含 field_name, field_comment,
                        table_name, sample_value。

        Returns:
            dict: 分类结果，包含 level, reason, matched_rules, references,
                  confidence, need_manual_review。
        """
        # 步骤 1: 向量检索
        evidence_list = self.retrieve_evidence(field_info, k=3)

        # 步骤 2: 组装提示词
        field_profile_json = json.dumps(ensure_ascii=False, indent=2)
        evidence_json = json.dumps(
            [e.model_dump() for e in evidence_list], ensure_ascii=False, indent=2
        )

        user_message = CLASSIFICATION_USER.format(
            field_profile=field_profile_json, evidence=evidence_json
        )

        # 步骤 3: LLM 推理
        try:
            output = self.llm_service.classify(user_message)

            return {
                "level": output.level,
                "reason": output.reason,
                "matched_rules": output.matched_rules,
                "references": output.references,
                "confidence": output.confidence,
                "need_manual_review": output.need_manual_review,
            }

        except Exception as e:
            logger.error("LLM 调用失败: %s", str(e))
            return {
                "level": '未知',
                "reason":  f"LLM 调用失败: {str(e)}",
                "matched_rules": [],
                "references": [],
                "confidence": 0.0,
                "need_manual_review": True,
            }

    def set_error_log_cache(self, cache: dict):
        """更新错题本缓存。"""
        self.error_cache.clear()
        self.error_cache.update(cache)
