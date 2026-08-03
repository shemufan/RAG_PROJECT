"""Rebuild the Chroma collection from tracked TXT and PDF knowledge documents."""

from pathlib import Path

from app.services.knowledge_service import KnowledgeService, load_knowledge_documents
from app.services.ocr_service import QwenOCRService


def load_documents(
    knowledge_dir: str | Path,
    *,
    version: str,
    ocr_service=None,
):
    """Load all configured knowledge sources before any vector-store mutation."""
    return load_knowledge_documents(
        knowledge_dir,
        version=version,
        ocr_service=ocr_service,
    )


def build_ocr_service(settings) -> QwenOCRService:
    """Build the configured Qwen OCR adapter only when PDF laws require it."""
    if not settings.qwen_ocr_api_key:
        raise RuntimeError("QWEN_OCR_API_KEY is required when PDF laws exist")
    if not settings.qwen_ocr_base_url:
        raise RuntimeError("QWEN_OCR_BASE_URL is required when PDF laws exist")
    return QwenOCRService(
        api_key=settings.qwen_ocr_api_key,
        base_url=settings.qwen_ocr_base_url,
        model=settings.qwen_ocr_model,
        cache_dir=settings.qwen_ocr_cache_dir,
        timeout_seconds=settings.qwen_ocr_timeout_seconds,
        max_retries=settings.qwen_ocr_max_retries,
    )


def rebuild_knowledge_base(settings, *, vector_store, ocr_service=None) -> int:
    """Load every source successfully, then atomically replace Chroma content."""
    laws_dir = settings.knowledge_dir / "laws"
    if any(laws_dir.glob("*.pdf")) and ocr_service is None:
        ocr_service = build_ocr_service(settings)
    documents = load_documents(
        settings.knowledge_dir,
        version=settings.knowledge_base_version,
        ocr_service=ocr_service,
    )
    return KnowledgeService(vector_store).rebuild(documents)


def main() -> None:
    from app.core.config import get_settings
    from app.repositories.vector_store import VectorStore
    from app.services.embedding_service import EmbeddingService

    settings = get_settings()
    store = VectorStore(
        EmbeddingService(model_path=settings.embedding_model_path),
        settings=settings,
    )
    count = rebuild_knowledge_base(settings, vector_store=store)
    print(f"知识库重建完成：{count} 个知识块")


if __name__ == "__main__":
    main()
