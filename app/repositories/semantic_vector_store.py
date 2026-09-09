"""Independent Chroma adapter for field-semantic cards."""

import json
from typing import Any

from app.schemas.semantic import SemanticCard, SemanticRetrievalResult


def _card_from_document(document: Any) -> SemanticCard:
    metadata = document.metadata or {}
    return SemanticCard(
        semantic_type=metadata["semantic_type"],
        aliases=json.loads(metadata["aliases_json"]),
        common_field_names=json.loads(metadata["common_field_names_json"]),
        value_features=json.loads(metadata["value_features_json"]),
        description=metadata["description"],
        semantic_category=json.loads(metadata["semantic_category_json"]),
        regulation_keywords=json.loads(metadata["regulation_keywords_json"]),
    )


class SemanticVectorStore:
    """Store Semantic Cards separately from regulatory documents."""

    def __init__(self, embedding_service=None, *, client=None, settings=None):
        if client is not None:
            self._store = client
            self._settings = settings
            self._embedding_function = None
            return
        from langchain_chroma import Chroma

        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()
        if embedding_service is None:
            raise ValueError("embedding_service 不能为空")
        self._settings = settings
        self._embedding_function = embedding_service.get_embeddings()
        self._store = Chroma(
            collection_name=settings.semantic_chroma_collection,
            persist_directory=str(settings.semantic_chroma_db_dir),
            embedding_function=self._embedding_function,
        )

    def search(self, query: str, k: int = 3) -> list[SemanticRetrievalResult]:
        rows = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [
            SemanticRetrievalResult(
                card=_card_from_document(document),
                raw_score=float(score),
            )
            for document, score in rows
        ]

    def add_documents(self, documents: list[Any]) -> list[str]:
        return self._store.add_documents(documents)

    def count(self) -> int:
        return self._store._collection.count()

    def reset(self) -> None:
        self._store.delete_collection()
        from langchain_chroma import Chroma

        self._store = Chroma(
            collection_name=self._settings.semantic_chroma_collection,
            persist_directory=str(self._settings.semantic_chroma_db_dir),
            embedding_function=self._embedding_function,
        )

