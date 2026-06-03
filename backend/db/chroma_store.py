# backend/db/chroma_store.py
"""Chroma 向量库操作 — 文档检索、入库、更新。"""

import os
import logging
from langchain_chroma import Chroma
from backend.core.config import DB_PATH
from backend.utils.chunker import split_by_article

logger = logging.getLogger(__name__)


class ChromaStore:
    """Chroma 向量库封装：相似性检索 + 知识库更新。

    依赖 EmbeddingService 提供的 embedding 函数，
    在 __init__ 中绑定到 Chroma 持久化目录。
    """

    def __init__(self, embedding_service):
        self.db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_service.get_embeddings(),
        )

    def similarity_search(self, query: str, k: int = 3):
        """检索与查询最相关的 top-k 文档 chunk。

        Args:
            query: 语义搜索查询字符串。
            k: 返回的文档数量。

        Returns:
            list[Document]: LangChain Document 列表。
        """
        return self.db.similarity_search(query, k=k)

    def add_documents(self, documents: list):
        """批量添加文档到向量库。

        Args:
            documents: LangChain Document 列表。

        Returns:
            list: 分配的文档 ID 列表。
        """
        return self.db.add_documents(documents)

    def update_knowledge_base(self, file_objs) -> str:
        """处理上传的法规文件，分块后入库。

        对应原 engine.py 的 update_knowledge_base() 函数。

        Args:
            file_objs: Gradio File 组件上传的文件对象列表。

        Returns:
            str: 给前端的状态提示信息。
        """
        if not file_objs:
            return " 未选择文件。"
        try:
            # 先处理所有文件的分块，全部成功后再一次性写入
            all_chunks = []
            for file_obj in file_objs:
                file_name = os.path.basename(file_obj.name)

                with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                chunks = split_by_article(
                    text=text,
                    document_name=file_name,
                    source_type="enterprise_rule",
                    domain="general",
                )
                all_chunks.extend(chunks)

            if all_chunks:
                self.db.add_documents(all_chunks)
                self.db.persist()

            return f" 知识库更新成功！新增 {len(all_chunks)} 个知识区块并已持久化到本地。"

        except Exception as e:
            return f" 更新失败：{str(e)}"
