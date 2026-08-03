"""Load and rebuild the regulatory knowledge collection."""

import logging
from pathlib import Path

from app.rag.chunker import split_knowledge_text

logger = logging.getLogger(__name__)


def read_source_text(path: Path) -> str:
    """Read a tracked text source with the supported legacy fallback."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s 使用 GB18030 解码", path.name)
        return raw.decode("gb18030")


def _law_paths(laws_dir: Path) -> list[Path]:
    paths = [*laws_dir.glob("*.txt"), *laws_dir.glob("*.pdf")]
    ordered = sorted(paths, key=lambda item: item.name.casefold())
    stems: dict[str, Path] = {}
    for path in ordered:
        identity = path.stem.casefold()
        previous = stems.get(identity)
        if previous is not None:
            raise RuntimeError(
                f"{path.stem}: duplicate law sources {previous.name} and {path.name}"
            )
        stems[identity] = path
    return ordered


def load_knowledge_documents(
    knowledge_dir: str | Path,
    *,
    version: str,
    ocr_service=None,
) -> list:
    """Load classification rules plus TXT/PDF laws into structured chunks."""
    root = Path(knowledge_dir)
    sources: list[tuple[Path, str]] = []
    rules_path = root / "classification_rules.md"
    if rules_path.is_file():
        sources.append((rules_path, "classification_rule"))
    sources.extend((path, "legal_document") for path in _law_paths(root / "laws"))

    documents = []
    for path, source_type in sources:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if ocr_service is None:
                raise RuntimeError(f"{path.name}: OCR service is required for PDF laws")
            text = ocr_service.extract_pdf(path)
        else:
            text = read_source_text(path)
        documents.extend(
            split_knowledge_text(
                text,
                path.name,
                source_type=source_type,
                source_format=suffix.removeprefix("."),
                version=version,
            )
        )
    if not documents:
        raise RuntimeError(f"未在 {root} 找到知识文档")
    return documents


class KnowledgeService:
    """Replace the vector collection with fully loaded knowledge chunks."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def rebuild(self, documents: list) -> int:
        self.vector_store.reset()
        self.vector_store.add_documents(documents)
        return len(documents)
