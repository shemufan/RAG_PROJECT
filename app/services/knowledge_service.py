"""Prepare validated knowledge documents and ingest a candidate collection."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunker import split_knowledge_pages
from app.schemas.knowledge_quality import (
    CleanedKnowledge,
    ExtractedPage,
    KnowledgeQualityReport,
    SourceManifest,
)
from app.services.knowledge_cleaner import KnowledgeCleaner
from app.services.knowledge_quality_service import KnowledgeQualityService
from app.services.ocr_service import OCR_PROMPT_VERSION

logger = logging.getLogger(__name__)


class KnowledgeQualityError(RuntimeError):
    """Raised when one or more sources are unsafe to ingest."""


@dataclass(frozen=True)
class PreparedKnowledge:
    """Documents and auditable intermediate artifacts from one preparation run."""

    documents: list[Document]
    reports: list[KnowledgeQualityReport]
    manifests: list[SourceManifest]
    cleaned_sources: dict[str, CleanedKnowledge]


def read_source_text(path: Path) -> str:
    """Read a tracked text source with the supported legacy fallback."""
    text, _ = _read_source_text_with_encoding(path)
    return text


def _read_source_text_with_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        logger.warning("%s 使用 GB18030 解码", path.name)
        return raw.decode("gb18030"), "gb18030"


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


def _source_pages(path: Path, *, ocr_service) -> tuple[list[ExtractedPage], str, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if ocr_service is None:
            raise RuntimeError(f"{path.name}: OCR service is required for PDF laws")
        if hasattr(ocr_service, "extract_pdf_pages"):
            return ocr_service.extract_pdf_pages(path), OCR_PROMPT_VERSION, None
        text = ocr_service.extract_pdf(path)
        page = ExtractedPage(page_number=1, text=text, extraction_method="ocr")
        return [page], "legacy-ocr", None
    text, encoding = _read_source_text_with_encoding(path)
    page = ExtractedPage(page_number=1, text=text, extraction_method="text")
    return [page], "text-decoder-v1", encoding


def prepare_knowledge_documents(
    knowledge_dir: str | Path,
    *,
    version: str,
    ocr_service=None,
    cleaner: KnowledgeCleaner | None = None,
    quality_service: KnowledgeQualityService | None = None,
) -> PreparedKnowledge:
    """Extract, clean, validate, and split every configured knowledge source."""
    root = Path(knowledge_dir)
    sources: list[tuple[Path, str]] = []
    rules_path = root / "classification_rules.md"
    if rules_path.is_file():
        sources.append((rules_path, "classification_rule"))
    sources.extend((path, "legal_document") for path in _law_paths(root / "laws"))
    if not sources:
        raise RuntimeError(f"未在 {root} 找到知识文档")

    cleaner = cleaner or KnowledgeCleaner()
    quality_service = quality_service or KnowledgeQualityService()
    documents: list[Document] = []
    reports = []
    manifests = []
    cleaned_sources = {}
    for path, source_type in sources:
        pages, extractor_version, encoding = _source_pages(path, ocr_service=ocr_service)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = SourceManifest(
            document_name=path.name,
            source_format=path.suffix.lower().removeprefix("."),
            source_sha256=source_sha256,
            page_count=len(pages),
            extractor_version=extractor_version,
            encoding=encoding,
        )
        cleaned = cleaner.clean(pages)
        report = quality_service.evaluate(
            manifest,
            cleaned,
            requires_start_section=source_type == "legal_document",
        )
        chunks = split_knowledge_pages(
            cleaned.pages,
            path.name,
            source_type=source_type,
            source_format=manifest.source_format,
            version=version,
            source_sha256=source_sha256,
        )
        documents.extend(chunks)
        reports.append(report)
        manifests.append(manifest)
        cleaned_sources[path.name] = cleaned
    return PreparedKnowledge(
        documents=documents,
        reports=reports,
        manifests=manifests,
        cleaned_sources=cleaned_sources,
    )


def load_knowledge_documents(
    knowledge_dir: str | Path,
    *,
    version: str,
    ocr_service=None,
    approved_hashes: set[str] | None = None,
) -> list[Document]:
    """Return chunks only when every source passes the quality gate."""
    quality_service = KnowledgeQualityService()
    prepared = prepare_knowledge_documents(
        knowledge_dir,
        version=version,
        ocr_service=ocr_service,
        quality_service=quality_service,
    )
    rejected = [
        report
        for report in prepared.reports
        if not quality_service.can_ingest(report, approved_hashes=approved_hashes)
    ]
    if rejected:
        details = "; ".join(
            f"{report.document_name}: {report.status.value} ({', '.join(report.issues)})"
            for report in rejected
        )
        raise KnowledgeQualityError(details)
    return prepared.documents


class KnowledgeService:
    """Write fully prepared documents into an empty candidate collection."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def rebuild(self, documents: list[Document]) -> int:
        if self.vector_store.count() != 0:
            raise RuntimeError("candidate collection must be empty")
        self.vector_store.add_documents(documents)
        return len(documents)
