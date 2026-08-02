"""Rebuild the Chroma collection from tracked knowledge documents."""

from pathlib import Path

from app.services.knowledge_service import (
    KnowledgeService,
    load_knowledge_documents,
)


def load_documents(knowledge_dir: str | Path, *, version: str):
    return load_knowledge_documents(knowledge_dir, version=version)


def main() -> None:
    from app.core.config import get_settings
    from app.repositories.vector_store import VectorStore
    from app.services.embedding_service import EmbeddingService

    settings = get_settings()
    store = VectorStore(
        EmbeddingService(model_path=settings.embedding_model_path),
        settings=settings,
    )
    documents = load_documents(
        settings.knowledge_dir,
        version=settings.knowledge_base_version,
    )
    count = KnowledgeService(store).rebuild(documents)
    print(f"知识库重建完成：{count} 个知识块")


if __name__ == "__main__":
    main()
