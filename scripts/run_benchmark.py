"""Run or resume a labeled two-file CSV benchmark directly to database B."""

import argparse
from collections.abc import Sequence
from uuid import UUID

from app.core.config import load_settings
from app.repositories.benchmark_target import BenchmarkTargetRepository
from app.repositories.vector_store import VectorStore
from app.schemas.benchmark import BenchmarkRunSummary
from app.services.benchmark_label_service import (
    BenchmarkCSVInput,
    prepare_labeled_catalog_benchmark,
)
from app.services.classification_service import FieldClassificationService
from app.services.csv_pipeline import CSVClassificationPipeline
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personal", required=True, help="personal-field catalog CSV")
    parser.add_argument(
        "--non-personal",
        required=True,
        help="non-personal-field catalog CSV",
    )
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--batch", help="start a new run for this imported batch")
    run_mode.add_argument("--resume-run", type=UUID, help="resume an existing run ID")
    parser.add_argument("--personal-limit", type=_positive_integer)
    parser.add_argument("--non-personal-limit", type=_positive_integer)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="when resuming, retry failed cases but never successful cases",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume_run is not None and (
        args.personal_limit is not None or args.non_personal_limit is not None
    ):
        parser.error("limits belong to a new --batch run and cannot change a resume")
    if args.batch is not None and args.retry_failed:
        parser.error("--retry-failed requires --resume-run")
    return args


def load_benchmark_input(
    args: argparse.Namespace,
    *,
    resume_summary: BenchmarkRunSummary | None = None,
) -> BenchmarkCSVInput:
    if args.resume_run is not None and resume_summary is None:
        raise ValueError("resume summary is required to rebuild the original input")
    batch_name = resume_summary.batch_name if resume_summary else args.batch
    personal_limit = (
        resume_summary.personal_limit if resume_summary else args.personal_limit
    )
    non_personal_limit = (
        resume_summary.non_personal_limit if resume_summary else args.non_personal_limit
    )
    return prepare_labeled_catalog_benchmark(
        args.personal,
        args.non_personal,
        batch_name=batch_name,
        personal_limit=personal_limit,
        non_personal_limit=non_personal_limit,
    )


def build_pipeline(settings) -> CSVClassificationPipeline:
    if not settings.target_database_url:
        raise SystemExit("TARGET_DATABASE_URL is not configured")
    embedding_service = EmbeddingService(model_path=settings.embedding_model_path)
    vector_store = VectorStore(embedding_service, settings=settings)
    if vector_store.count() == 0:
        raise SystemExit(
            "knowledge base is empty; run python -m scripts.rebuild_knowledge_base"
        )
    classifier = FieldClassificationService(
        vector_store,
        LLMService(settings=settings),
    )
    return CSVClassificationPipeline(
        BenchmarkTargetRepository(settings.target_database_url),
        classifier,
        model_name=settings.deepseek_model,
        knowledge_base_version=settings.knowledge_base_version,
    )


def print_summary(summary: BenchmarkRunSummary) -> None:
    print(f"run_id={summary.run_id}")
    print(
        "cases: "
        f"total={summary.total_cases}, success={summary.success_cases}, "
        f"failed={summary.failed_cases}"
    )
    print(f"confusion: TP={summary.tp}, FP={summary.fp}, TN={summary.tn}, FN={summary.fn}")
    print(
        "scores: "
        f"precision={summary.precision_score:.4f}, "
        f"recall={summary.recall_score:.4f}, "
        f"f1={summary.f1_score:.4f}, "
        f"accuracy={summary.accuracy_score:.4f}, "
        f"coverage={summary.coverage_score:.4f}, "
        f"effective_recall={summary.effective_recall_score:.4f}"
    )
    print(f"status={summary.status}")


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(load_settings())
    resume_summary = None
    if args.resume_run is not None:
        resume_summary = pipeline.target_repository.get_run(args.resume_run)
        if resume_summary is None:
            raise SystemExit("benchmark run not found")
    prepared = load_benchmark_input(args, resume_summary=resume_summary)
    if args.resume_run is not None:
        summary = pipeline.resume(
            args.resume_run,
            prepared.batch,
            prepared.labels,
            retry_failed=args.retry_failed,
        )
    else:
        summary = pipeline.run(prepared.batch, prepared.labels)
    print_summary(summary)


if __name__ == "__main__":
    main()
