# core/engine.py
import logging
import os
import re

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from core.config import MODEL_PATH, DB_PATH  # 引入刚才定义的配置

logger.info("正在加载本地知识库与模型，请稍候...")

# 1. 全局初始化 (只会在 main.py 启动时执行一次)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH, model_kwargs={"local_files_only": True})
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

prompt_template = PromptTemplate(
    input_variables=["context", "field_en", "field_cn", "desc"],
    template="""你是一个数据合规专家。请根据以下法律条文：
    {context}

    分析以下业务字段：
    英文名：{field_en}
    中文名：{field_cn}
    描述：{desc}

    请严格按照 Step 1, Step 2, Step 3 进行分析，并在最后一行以「最终定级：LX」的格式明确输出级别（L1/L2/L3/L4 四选一）。""",
)

# 跨模块共享变量（通过 set_error_log_cache 写入，禁止外部直接修改）
ERROR_LOG_CACHE = {}


def set_error_log_cache(cache: dict):
    """安全写入错题本缓存，避免跨模块直接操作全局变量。"""
    ERROR_LOG_CACHE.clear()
    ERROR_LOG_CACHE.update(cache)


def update_knowledge_base(file_objs):
    if not file_objs:
        return " 未选择文件。"
    try:
        total_chunks = 0
        for file_obj in file_objs:
            with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.create_documents([text])
            vector_db.add_documents(chunks)
            total_chunks += len(chunks)
        vector_db.persist()
        return f" 知识库更新成功！新增 {total_chunks} 个知识区块并已持久化到本地。"
    except Exception as e:
        return f" 更新失败：{str(e)}"


def smart_predict(field_en, field_cn, desc):
    if field_en in ERROR_LOG_CACHE:
        return (
            ERROR_LOG_CACHE[field_en],
            "🎯 [人工干预] 触发错题本强规则拦截，跳过大模型。",
            "⚠️ 命中强制拦截规则，无需检索法律依据。",
        )

    if vector_db._collection.count() == 0:
        return "未知", "⚠️ 知识库为空，请先上传法规文本初始化知识库。", "知识库中无任何法律依据。"

    search_query = f"{field_cn} {desc}"
    docs = vector_db.similarity_search(search_query, k=3)  # 从知识库中选取3条最相关的依据

    context_list = []
    for i, doc in enumerate(docs):
        context_list.append(f"【依据 {i+1}】: {doc.page_content}")
    retrieved_context_all = "\n\n".join(context_list)

    retrieved_context = "\n".join([doc.page_content for doc in docs])

    final_prompt = prompt_template.format(
        context=retrieved_context, field_en=field_en, field_cn=field_cn, desc=desc
    )

    try:
        # 调用大模型
        response = llm.invoke(final_prompt).content

        level = "未知"
        # 优先匹配「最终定级：LX」格式，防止误匹配正文中的级别提及
        m = re.search(r"最终定级[：:]\s*(L[1-4])", response)
        if m:
            level = m.group(1)
        else:
            # 回退：取全文最后一次出现的 L1-L4
            matches = re.findall(r"\b(L[1-4])\b", response)
            if matches:
                level = matches[-1]

        return (
            level,
            response,
            retrieved_context_all,
        )  # 返回级别和大模型的完整推理过程以及检索到的法律依据
    except Exception as e:
        return "Error", f" 调用大模型出错: {str(e)}"
