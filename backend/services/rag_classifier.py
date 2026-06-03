# backend/services/rag_classifier.py
"""RAG 分类主流程 — 错题本拦截 → 向量检索 → LLM 判级。

这是整个系统的核心管线，整合了 ChromaStore (检索) 和 LLMService (推理)。
对应原 core/engine.py 的 classify_field()、retrieve_evidence()、
build_search_query()、smart_predict()、set_error_log_cache() 和 ERROR_LOG_CACHE。
"""

import json
import logging
from backend.schemas.classify_schema import FieldProfile, Evidence, ClassificationResult
from backend.services.prompt import CLASSIFICATION_USER

logger = logging.getLogger(__name__)


class RAGClassifier:
    """RAG 分类管线：错题本缓存 → 向量检索 → LLM 推理。

    实例属性 error_cache 替代了原来 engine.py 的模块级全局变量 ERROR_LOG_CACHE，
    使状态显式化，支持测试隔离和多实例部署。

    依赖:
        chroma_store: ChromaStore 实例（向量检索）
        llm_service:  LLMService 实例（LLM 推理）
        error_cache:  dict，字段英文名 → 强制分类级别（错题本缓存）
    """

    def __init__(self, chroma_store, llm_service, error_cache=None):
        self.chroma_store = chroma_store
        self.llm_service = llm_service
        self.error_cache = error_cache if error_cache is not None else {}

    # ── 错题本缓存管理 ──────────────────────────────────────

    # 合法的敏感等级值集合（与 ClassificationResult.level Literal 一致）
    _VALID_LEVELS = frozenset({"L1", "L2", "L3", "L4", "未知"})

    def set_error_log_cache(self, cache: dict):
        """替换错题本缓存（安全写入，禁止外部直接操作内部 dict）。

        对应原 engine.py 的 set_error_log_cache()。

        Args:
            cache: 字段英文名 → 标准答案级别的映射字典。
        """
        self.error_cache.clear()
        self.error_cache.update(cache)

    # ── 检索阶段 ────────────────────────────────────────────

    def build_search_query(self, field: FieldProfile) -> str:
        """将 FieldProfile 拼接为语义搜索查询字符串。

        对应原 engine.py 的 build_search_query()。
        """
        return f"""
字段名：{field.field_name}
中文名：{field.field_cn or ""}
字段注释：{field.field_comment or ""}
字段类型：{field.data_type or ""}
表名：{field.table_name or ""}
样例值：{", ".join(field.sample_values[:5])}
业务域：{field.business_domain}
"""

    def retrieve_evidence(self, field: FieldProfile, k: int = 3) -> list[Evidence]:
        """向量检索，返回 top-k 法规依据。

        对应原 engine.py 的 retrieve_evidence()。
        """
        query = self.build_search_query(field)

        # Layer1 阶段先不用复杂过滤，后续可加 where
        docs = self.chroma_store.similarity_search(query, k=k)

        evidence_list = []
        for idx, doc in enumerate(docs):
            metadata = doc.metadata or {}
            evidence_list.append(
                Evidence(
                    document_name=metadata.get("document_name", "未知文档"),
                    source_type=metadata.get("source_type", "unknown"),
                    domain=metadata.get("domain", "general"),
                    hierarchy_level=metadata.get("hierarchy_level"),
                    content=doc.page_content,
                    score=None,
                    chunk_id=metadata.get("chunk_id", f"evidence_{idx+1}"),
                )
            )

        return evidence_list

    # ── 分类主流程 ──────────────────────────────────────────

    def classify_field(self, field: FieldProfile) -> ClassificationResult:
        """RAG 分类主流程：错题本检查 → 检索 → LLM 推理。

        对应原 engine.py 的 classify_field()，逻辑完全保留。

        Args:
            field: 待分类的字段画像。

        Returns:
            ClassificationResult: 包含分类结论、依据、决策路径的完整结果。
        """
        # 步骤 0: 错题本拦截（命中则跳过 RAG + LLM）
        if field.field_name in self.error_cache:
            forced_level = self.error_cache[field.field_name]
            # 校验缓存中的 level 值合法性，非法值回退到 RAG+LLM 流程
            if forced_level not in self._VALID_LEVELS:
                logger.warning(
                    "错题本缓存中存在非法 level 值 '%s' (字段: %s)，回退到 RAG+LLM 流程",
                    forced_level, field.field_name,
                )
            else:
                return ClassificationResult(
                    field_name=field.field_name,
                    database_name=field.database_name,
                    table_name=field.table_name,
                    category="人工纠错规则",
                    subcategory="错题本强制拦截",
                    level=forced_level,
                    confidence=1.0,
                    reason="命中人工上传的错题本规则，跳过 RAG 和 LLM。",
                    evidence=[],
                    need_review=False,
                    decision_path="error_log_cache",
                    raw_response=None,
                )

        # 步骤 1: 向量检索
        evidence_list = self.retrieve_evidence(field, k=3)

        # 步骤 2: 组装提示词
        field_profile_json = field.model_dump_json(ensure_ascii=False, indent=2)
        evidence_json = json.dumps(
            [e.model_dump() for e in evidence_list], ensure_ascii=False, indent=2
        )

        user_message = CLASSIFICATION_USER.format(
            field_profile=field_profile_json, evidence=evidence_json
        )

        # 步骤 3: LLM 推理
        try:
            output = self.llm_service.classify(user_message)

            return ClassificationResult(
                field_name=field.field_name,
                database_name=field.database_name,
                table_name=field.table_name,
                category=output.category,
                subcategory=output.subcategory,
                level=output.level,
                confidence=output.confidence,
                reason=output.reason,
                evidence=evidence_list,
                need_review=output.need_review,
                decision_path="rag_llm",
                raw_response=output.model_dump_json(ensure_ascii=False),
            )

        except Exception as e:
            logger.error("LLM 调用失败: %s", str(e))
            return ClassificationResult(
                field_name=field.field_name,
                database_name=field.database_name,
                table_name=field.table_name,
                category="未知",
                subcategory=None,
                level="未知",
                confidence=0.0,
                reason=f"LLM 调用失败: {str(e)}",
                evidence=evidence_list,
                need_review=True,
                decision_path="rag_llm_error",
                raw_response=None,
            )

    # ── 便捷方法 ─────────────────────────────────────────────

    def smart_predict(self, field_en: str, field_cn: str, desc: str):
        """单字段分类便捷接口（Gradio 单条测试专用）。

        对应原 engine.py 的 smart_predict()。

        Args:
            field_en: 字段英文名。
            field_cn: 字段中文名。
            desc:     业务描述。

        Returns:
            (level, reason_text, evidence_text): Gradio 三输出组件的值。
        """
        field = FieldProfile(
            field_name=field_en,
            field_cn=field_cn,
            field_comment=desc,
            business_domain="general",
        )

        result = self.classify_field(field)

        evidence_text = "\n\n".join(
            [
                f"【依据 {i+1}】{e.document_name} {e.hierarchy_level or ''}\n{e.content}"
                for i, e in enumerate(result.evidence)
            ]
        )

        reason_text = f"""
分类：{result.category}
细分类：{result.subcategory}
等级：{result.level}
置信度：{result.confidence}
是否复核：{result.need_review}
理由：{result.reason}
"""

        return result.level, reason_text, evidence_text
