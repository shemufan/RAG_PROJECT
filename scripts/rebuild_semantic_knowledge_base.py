"""Rebuild the independent Semantic Knowledge Base collection."""

from pathlib import Path

from app.services.semantic_knowledge_service import (
    SemanticKnowledgeService,
    load_semantic_documents,
)


def rebuild_semantic_knowledge_base(
    semantic_knowledge_file: str | Path,
    *,
    vector_store,
) -> int:
    documents = load_semantic_documents(semantic_knowledge_file)
    return SemanticKnowledgeService(vector_store).rebuild(documents)


def main() -> None:
    from app.core.config import get_settings
    from app.repositories.semantic_vector_store import SemanticVectorStore
    from app.services.embedding_service import EmbeddingService

    settings = get_settings()
    store = SemanticVectorStore(
        EmbeddingService(model_path=settings.embedding_model_path),
        settings=settings,
    )
    count = rebuild_semantic_knowledge_base(
        settings.semantic_knowledge_file,
        vector_store=store,
    )
    print(f"Semantic KB 重建完成：{count} 张语义卡片")


if __name__ == "__main__":
    main()
