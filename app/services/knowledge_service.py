"""Load and rebuild the regulatory knowledge collection."""

import logging
from pathlib import Path

from app.rag.chunker import split_knowledge_text

logger = logging.getLogger(__name__)


def read_source_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s 使用 GB18030 解码", path.name)
        return raw.decode("gb18030")


def load_knowledge_documents(
    knowledge_dir: str | Path,
    *,
    version: str,
) -> list:
    """Load classification rules and law texts into structured chunks."""
    root = Path(knowledge_dir)
    sources = [(root / "classification_rules.md", "classification_rule")]
    sources.extend((path, "legal_document") for path in sorted((root / "laws").glob("*.txt")))
    documents = []
    for path, source_type in sources:
        if not path.is_file():
            continue
        documents.extend(
            split_knowledge_text(
                read_source_text(path),
                path.name,
                source_type=source_type,
                version=version,
            )
        )
    if not documents:
        raise RuntimeError(f"未在 {root} 找到知识文档")
    return documents


class KnowledgeService:
    """Replace the vector collection with chunks loaded from disk."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def rebuild(self, documents: list) -> int:
        self.vector_store.reset()
        self.vector_store.add_documents(documents)
        return len(documents)
