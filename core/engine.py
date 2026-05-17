# core/engine.py
import logging
import os
import re
import json
from core.schemas import FieldProfile, Evidence, ClassificationResult
from core.prompt import CLASSIFICATION_PROMPT
from knowledge.chunker import split_by_article

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.config import MODEL_PATH, DB_PATH  # 引入刚才定义的配置

logger.info("正在加载本地知识库与模型，请稍候...")


# 解析JSON
def parse_llm_json_response(response: str) -> dict:
    """
    尽量从 LLM 输出中解析 JSON。
    即使模型误包了 Markdown，也尽量提取。
    """
    try:
        return json.loads(response)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"无法解析 LLM JSON 输出: {response}")


def build_search_query(field: FieldProfile) -> str:
    return f"""
字段名：{field.field_name}
中文名：{field.field_cn or ""}
字段注释：{field.field_comment or ""}
字段类型：{field.data_type or ""}
表名：{field.table_name or ""}
样例值：{", ".join(field.sample_values[:5])}
业务域：{field.business_domain}
"""


def retrieve_evidence(field: FieldProfile, k: int = 3) -> list[Evidence]:
    query = build_search_query(field)

    # Layer1 阶段先不用复杂过滤，后续可加 where
    docs = vector_db.similarity_search(query, k=k)

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


# 1. 全局初始化 (只会在 main.py 启动时执行一次)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH, model_kwargs={"local_files_only": True})
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
# 跨模块共享变量（通过 set_error_log_cache 写入，禁止外部直接修改）
ERROR_LOG_CACHE = {}


def set_error_log_cache(cache: dict):
    """安全写入错题本缓存，避免跨模块直接操作全局变量。

    Args:
        cache: 字段英文名 → 标准答案级别的映射字典，如 {"salary": "L3", "account_code": "L2"}
    """
    ERROR_LOG_CACHE.clear()
    ERROR_LOG_CACHE.update(cache)


def update_knowledge_base(file_objs):
    if not file_objs:
        return " 未选择文件。"
    try:
        total_chunks = 0

        for file_obj in file_objs:
            file_name = os.path.basename(file_obj.name)

            with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = split_by_article(
                text=text, document_name=file_name, source_type="enterprise_rule", domain="general"
            )

            vector_db.add_documents(chunks)
            total_chunks += len(chunks)

        vector_db.persist()
        return f" 知识库更新成功！新增 {total_chunks} 个知识区块并已持久化到本地。"

    except Exception as e:
        return f" 更新失败：{str(e)}"


def classify_field(field: FieldProfile) -> ClassificationResult:

    evidence_list = retrieve_evidence(field, k=3)

    field_profile_json = field.model_dump_json(ensure_ascii=False, indent=2)
    evidence_json = json.dumps(
        [e.model_dump() for e in evidence_list], ensure_ascii=False, indent=2
    )

    final_prompt = CLASSIFICATION_PROMPT.format(
        field_profile=field_profile_json, evidence=evidence_json
    )

    response = llm.invoke(final_prompt).content

    try:
        parsed = parse_llm_json_response(response)

        result = ClassificationResult(
            field_name=field.field_name,
            category=parsed.get("category", "未知"),
            subcategory=parsed.get("subcategory"),
            level=parsed.get("level", "未知"),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=parsed.get("reason", ""),
            evidence=evidence_list,
            need_review=bool(parsed.get("need_review", False)),
            decision_path="rag_llm",
            raw_response=response,
        )
        return result

    except Exception as e:
        return ClassificationResult(
            field_name=field.field_name,
            category="未知",
            level="未知",
            confidence=0.0,
            reason=f"LLM 输出解析失败: {str(e)}",
            evidence=evidence_list,
            need_review=True,
            decision_path="rag_llm_error",
            raw_response=response,
        )


def smart_predict(field_en, field_cn, desc):
    field = FieldProfile(
        field_name=field_en, field_cn=field_cn, field_comment=desc, business_domain="general"
    )

    result = classify_field(field)

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
