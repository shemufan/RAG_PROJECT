"""Chroma adapter for knowledge retrieval and persistence."""

from typing import Any

from app.schemas.classification import Evidence


def map_retrieved_document(document: Any, score: float | None) -> Evidence:
    """Map a LangChain document and relevance score to business evidence."""
    metadata = document.metadata or {}
    normalized_score = None
    if score is not None:
        normalized_score = max(0.0, min(1.0, float(score)))
    return Evidence(
        content=document.page_content,
        source=metadata.get("document_name") or metadata.get("source") or "未知来源",
        article=metadata.get("article") or metadata.get("hierarchy_level"),
        score=normalized_score,
        chunk_id=metadata.get("chunk_id"),
    )


class VectorStore:
    """Store and retrieve regulatory knowledge in one Chroma collection."""

    def __init__(self, embedding_service=None, *, client=None, settings=None, collection_name=None):
        if client is not None:
            self._store = client
            return

        from langchain_chroma import Chroma

        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()
        if embedding_service is None:
            raise ValueError("embedding_service 不能为空")
        self._store = Chroma(
            collection_name=collection_name or settings.chroma_collection,
            persist_directory=str(settings.chroma_db_dir),
            embedding_function=embedding_service.get_embeddings(),
        )

    def search(self, query: str, k: int = 3) -> list[Evidence]:
        rows = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [map_retrieved_document(document, score) for document, score in rows]


    def add_documents(self, documents: list[Any]) -> list[str]:
        return self._store.add_documents(documents)

    def count(self) -> int:
        return self._store._collection.count()
