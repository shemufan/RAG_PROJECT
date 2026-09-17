"""Check knowledge quality and build an isolated candidate Chroma collection."""

import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median

from app.services.knowledge_quality_service import KnowledgeQualityService
from app.services.knowledge_service import (
    KnowledgeService,
    PreparedKnowledge,
    load_knowledge_documents,
    prepare_knowledge_documents,
)
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


def rebuild_knowledge_base(
    settings,
    *,
    vector_store,
    ocr_service=None,
    approved_hashes: set[str] | None = None,
) -> int:
    """Load validated sources and populate an empty candidate store."""
    laws_dir = settings.knowledge_dir / "laws"
    if any(laws_dir.glob("*.pdf")) and ocr_service is None:
        ocr_service = build_ocr_service(settings)
    documents = load_knowledge_documents(
        settings.knowledge_dir,
        version=settings.knowledge_base_version,
        ocr_service=ocr_service,
        approved_hashes=approved_hashes,
    )
    return KnowledgeService(vector_store).rebuild(documents)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查并重建法规知识库")
    parser.add_argument("--check-only", action="store_true", help="只检查，不连接 Chroma")
    parser.add_argument("--candidate-collection", help="候选 Chroma Collection 名称")
    parser.add_argument("--report-dir", type=Path, help="质量报告根目录")
    parser.add_argument(
        "--approve-review",
        action="append",
        default=[],
        metavar="SHA256",
        help="批准一个 REVIEW 文档的源文件 SHA-256；可重复传入",
    )
    return parser


def _default_report_dir(settings) -> Path:
    project_root = getattr(settings, "project_root", None)
    if project_root is None:
        project_root = Path(settings.knowledge_dir)
    return Path(project_root) / ".runtime" / "knowledge_quality"


def _write_quality_artifacts(
    prepared: PreparedKnowledge,
    *,
    report_dir: Path,
    version: str,
) -> Path:
    run_dir = report_dir / version
    run_dir.mkdir(parents=True, exist_ok=True)
    for manifest, report in zip(prepared.manifests, prepared.reports, strict=True):
        prefix = manifest.source_sha256[:12]
        (run_dir / f"{prefix}.manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / f"{prefix}.quality.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (run_dir / f"{prefix}.cleaned.txt").write_text(
            prepared.cleaned_sources[manifest.document_name].text,
            encoding="utf-8",
        )
    status_counts = Counter(report.status.value for report in prepared.reports)
    summary = {
        "source_count": len(prepared.manifests),
        "status_counts": dict(sorted(status_counts.items())),
        "failed_documents": [
            report.document_name
            for report in prepared.reports
            if report.status.value != "PASS"
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _write_import_report(prepared: PreparedKnowledge, run_dir: Path) -> None:
    lengths = sorted(len(document.page_content) for document in prepared.documents)
    if not lengths:
        raise RuntimeError("candidate collection has no knowledge chunks")
    if lengths[-1] > 2000:
        raise RuntimeError(f"knowledge chunk exceeds 2000 characters: {lengths[-1]}")
    report = {
        "source_count": len(prepared.manifests),
        "chunk_count": len(lengths),
        "minimum_chunk_characters": lengths[0],
        "median_chunk_characters": median(lengths),
        "p95_chunk_characters": lengths[max(0, ceil(len(lengths) * 0.95) - 1)],
        "maximum_chunk_characters": lengths[-1],
    }
    (run_dir / "import.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def execute(
    settings,
    *,
    check_only: bool = False,
    candidate_collection: str | None = None,
    report_dir: Path | None = None,
    approved_hashes: set[str] | None = None,
    ocr_service=None,
    vector_store_factory=None,
    embedding_service_factory=None,
) -> int:
    """Run a quality check or populate a separate candidate collection."""
    laws_dir = settings.knowledge_dir / "laws"
    if any(laws_dir.glob("*.pdf")) and ocr_service is None:
        ocr_service = build_ocr_service(settings)
    prepared = prepare_knowledge_documents(
        settings.knowledge_dir,
        version=settings.knowledge_base_version,
        ocr_service=ocr_service,
    )
    run_dir = _write_quality_artifacts(
        prepared,
        report_dir=report_dir or _default_report_dir(settings),
        version=settings.knowledge_base_version,
    )
    quality_service = KnowledgeQualityService()
    rejected = [
        report
        for report in prepared.reports
        if not quality_service.can_ingest(
            report,
            approved_hashes=approved_hashes,
        )
    ]
    for report in prepared.reports:
        detail = f"：{'；'.join(report.issues)}" if report.issues else ""
        print(f"{report.status.value} {report.document_name}{detail}")
    print(f"质量报告：{run_dir}")
    if rejected:
        return 1
    if check_only:
        return 0

    collection = candidate_collection or (
        f"{settings.chroma_collection}__{settings.knowledge_base_version}"
    )
    if collection == settings.chroma_collection:
        raise RuntimeError("candidate collection must differ from current collection")
    if embedding_service_factory is None:
        from app.services.embedding_service import EmbeddingService

        embedding_service_factory = EmbeddingService
    if vector_store_factory is None:
        from app.repositories.vector_store import VectorStore

        vector_store_factory = VectorStore
    embeddings = embedding_service_factory(model_path=settings.embedding_model_path)
    store = vector_store_factory(
        embeddings,
        settings=settings,
        collection_name=collection,
    )
    KnowledgeService(store).rebuild(prepared.documents)
    _write_import_report(prepared, run_dir)
    print(f"候选知识库重建完成：{collection}，{len(prepared.documents)} 个知识块")
    return 0


def main(argv: list[str] | None = None) -> None:
    from app.core.config import get_settings

    args = create_parser().parse_args(argv)
    exit_code = execute(
        get_settings(),
        check_only=args.check_only,
        candidate_collection=args.candidate_collection,
        report_dir=args.report_dir,
        approved_hashes=set(args.approve_review),
    )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
