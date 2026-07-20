# backend/storage/chroma_store.py
"""Chroma 向量库操作 — 文档检索、入库、更新。"""

import logging
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from backend.core.config import CHROMA_COLLECTION, DB_PATH

logger = logging.getLogger(__name__)


def _query_terms(query: str) -> set[str]:
    values = []
    for line in query.splitlines():
        values.append(line.split(":", 1)[-1].strip())
    terms = set()
    for value in values:
        if len(value) >= 2:
            terms.add(value.lower())
        terms.update(
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z][a-zA-Z0-9_]{1,}", value)
        )
    return terms


def lexical_score(query: str, content: str) -> float:
    """Return the fraction of explicit field-profile terms found in a chunk."""
    terms = _query_terms(query)
    if not terms:
        return 0.0
    lowered = content.lower()
    return sum(term in lowered for term in terms) / len(terms)


class ChromaStore:
    """Chroma 向量库封装：相似性检索 + 知识库更新。

    依赖 EmbeddingService 提供的 embedding 函数，
    在 __init__ 中绑定到 Chroma 持久化目录。
    """

    def __init__(self, embedding_service):
        self.embedding_function = embedding_service.get_embeddings()
        self.db = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=DB_PATH,
            embedding_function=self.embedding_function,
        )

    def count(self) -> int:
        """Return the number of chunks in the baseline collection."""
        return self.db._collection.count()

    def similarity_search(self, query: str, k: int = 3):
        """检索与查询最相关的 top-k 文档 chunk。

        Args:
            query: 语义搜索查询字符串。
            k: 返回的文档数量。

        Returns:
            list[Document]: LangChain Document 列表。
        """
        return self.db.similarity_search(query, k=k)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 3):
        """Hybrid retrieval: vector candidates plus exact field-term matches.

        The local MPNet model is useful for broad semantics but weak on some Chinese
        field names. The Week3 corpus is small, so a lightweight lexical pass makes
        explicit names such as ``身份证号`` deterministic without adding a search service.
        """
        vector_rows = self.db.similarity_search_with_score(query, k=max(k * 4, 12))
        ranked: dict[tuple[str, str], tuple[Document, float]] = {}
        for document, distance in vector_rows:
            score = 1.0 / (1.0 + max(0.0, float(distance)))
            key = (document.page_content, document.metadata.get("document_name", ""))
            ranked[key] = (document, score)

        payload = self.db.get(include=["documents", "metadatas"])
        for content, metadata in zip(payload.get("documents", []), payload.get("metadatas", [])):
            term_score = lexical_score(query, content)
            if term_score <= 0:
                continue
            document = Document(page_content=content, metadata=metadata or {})
            key = (content, document.metadata.get("document_name", ""))
            score = 0.75 + 0.25 * term_score
            previous = ranked.get(key)
            if previous is None or score > previous[1]:
                ranked[key] = (document, score)

        return sorted(ranked.values(), key=lambda item: item[1], reverse=True)[:k]

    def reset(self) -> None:
        """Delete only the dedicated Week3 collection and recreate it."""
        try:
            self.db.delete_collection()
        except ValueError:
            pass
        self.db = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=DB_PATH,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: list):
        """批量添加文档到向量库。

        Args:
            documents: LangChain Document 列表。

        Returns:
            list: 分配的文档 ID 列表。
        """
        return self.db.add_documents(documents)

