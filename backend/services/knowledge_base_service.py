"""Knowledge-source ingestion separated from the vector storage adapter."""

import logging
from pathlib import Path
from typing import Iterable

from backend.utils.chunker import split_by_article

logger = logging.getLogger(__name__)


def read_source_text(path: Path) -> str:
    """Decode UTF-8 sources and legacy GB18030 compliance documents."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s 不是 UTF-8，按 GB18030 转码后入库", path.name)
        return raw.decode("gb18030")


class KnowledgeBaseService:
    """Convert source files into chunks and hand them to a vector store."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def update_from_paths(
        self,
        paths: Iterable[str | Path],
        *,
        source_type: str = "enterprise_rule",
        domain: str = "general",
    ) -> int:
        chunks = []
        for value in paths:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"知识源文件不存在: {path}")
            chunks.extend(
                split_by_article(
                    text=read_source_text(path),
                    document_name=path.name,
                    source_type=source_type,
                    domain=domain,
                )
            )
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)
