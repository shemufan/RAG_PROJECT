"""Classify one catalog or ordinary business CSV directly without database A."""

import argparse
from collections.abc import Sequence
from uuid import UUID

from app.core.config import load_settings
from app.rag.retrieval_query import ProfileQueryMode, QueryStrategy
from app.repositories.benchmark_target import BenchmarkTargetRepository
from app.repositories.vector_store import VectorStore
from app.schemas.csv_input import CSVInputBatch, LabelMatchSummary
from app.services.benchmark_label_service import (
    attach_benchmark_labels,
    attach_embedded_labels,
)
from app.services.catalog_csv_adapter import CatalogCSVAdapter
from app.services.classification_service import FieldClassificationService
from app.services.csv_mode_detector import resolve_csv_mode
from app.services.csv_pipeline import CSVClassificationPipeline
from app.services.csv_reader import CSVInputError, CSVReader
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.tabular_csv_adapter import TabularCSVAdapter


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="one CSV input path")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "catalog", "tabular"),
        default="auto",
    )
    parser.add_argument("--labels", help="optional field_name label CSV")
    parser.add_argument(
        "--label-column",
        help="ground-truth label column embedded in the input CSV",
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        help="classify only the first N cases (new runs only)",
    )
    parser.add_argument("--field-name-column")
    parser.add_argument("--sample-columns", help="comma-separated catalog columns")
    parser.add_argument("--resume-run", type=UUID)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--query-strategy",
        choices=("legacy", "clean", "profile"),
        default="profile",
        help="retrieval Query strategy; use the same value when resuming a run",
    )
    parser.add_argument(
        "--query-mode",
        choices=("c", "c1", "c2"),
        default="c",
        help="profile Query submode: c=full, c1=features only, c2=candidates only",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retry_failed and args.resume_run is None:
        parser.error("--retry-failed requires --resume-run")
    if (args.field_name_column or args.sample_columns) and args.input_mode != "catalog":
        parser.error("catalog column mappings require --input-mode catalog")
    if args.label_column and args.labels:
        parser.error("--label-column and --labels are mutually exclusive")
    if args.label_column and args.input_mode == "tabular":
        parser.error("--label-column requires catalog input (one field per row)")
    if args.limit is not None and args.resume_run is not None:
        parser.error("--limit applies to a new run and cannot be used with --resume-run")
    if args.query_strategy != "profile" and args.query_mode != "c":
        parser.error("--query-mode c1/c2 requires --query-strategy profile")
    return args


def load_csv_batch(args: argparse.Namespace) -> CSVInputBatch:
    inspection = CSVReader().inspect(args.input)
    ignored = {args.label_column} if args.label_column else None
    mode = resolve_csv_mode(inspection.headers, args.input_mode, ignored_headers=ignored)
    if mode == "tabular":
        if args.label_column:
            raise CSVInputError(
                "--label-column requires catalog input (one field per row)"
            )
        return TabularCSVAdapter().load(args.input)
    sample_columns = None
    if args.sample_columns is not None:
        sample_columns = [
            column.strip() for column in args.sample_columns.split(",") if column.strip()
        ]
    return CatalogCSVAdapter().load(
        args.input,
        field_name_column=args.field_name_column,
        sample_columns=sample_columns,
    )


def load_labels(
    batch: CSVInputBatch,
    args: argparse.Namespace,
) -> LabelMatchSummary | None:
    if args.label_column:
        return attach_embedded_labels(batch, args.input, label_column=args.label_column)
    if args.labels:
        return attach_benchmark_labels(batch, args.labels)
    return None


def apply_limit(
    batch: CSVInputBatch,
    labels: LabelMatchSummary | None,
    limit: int | None,
) -> tuple[CSVInputBatch, LabelMatchSummary | None]:
    if limit is None:
        return batch, labels
    limited_batch = batch.model_copy(update={"cases": batch.cases[:limit]})
    if labels is None:
        return limited_batch, None
    cases = labels.cases[:limit]
    limited_labels = labels.model_copy(
        update={
            "cases": cases,
            "labeled_cases": sum(case.expected_personal is not None for case in cases),
            "unlabeled_cases": sum(case.expected_personal is None for case in cases),
        }
    )
    return limited_batch, limited_labels


def build_pipeline(
    settings,
    query_strategy: QueryStrategy = "profile",
    profile_query_mode: ProfileQueryMode = "c",
) -> CSVClassificationPipeline:
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
        query_strategy=query_strategy,
        profile_query_mode=profile_query_mode,
    )
    return CSVClassificationPipeline(
        BenchmarkTargetRepository(settings.target_database_url),
        classifier,
        model_name=settings.deepseek_model,
        knowledge_base_version=settings.knowledge_base_version,
    )


def _score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_summary(summary) -> None:
    print(f"run_id={summary.run_id}")
    print(
        f"source={summary.source_name}, mode={summary.input_mode}, "
        f"status={summary.status}"
    )
    print(
        f"cases: total={summary.total_cases}, success={summary.success_cases}, "
        f"failed={summary.failed_cases}, labeled={summary.labeled_cases}, "
        f"unlabeled={summary.unlabeled_cases}"
    )
    print(f"confusion: TP={summary.tp}, FP={summary.fp}, TN={summary.tn}, FN={summary.fn}")
    print(
        "scores: "
        f"precision={_score(summary.precision_score)}, "
        f"recall={_score(summary.recall_score)}, "
        f"f1={_score(summary.f1_score)}, "
        f"accuracy={_score(summary.accuracy_score)}, "
        f"coverage={summary.coverage_score:.4f}, "
        f"effective_recall={_score(summary.effective_recall_score)}"
    )


def main() -> None:
    args = parse_args()
    batch = load_csv_batch(args)
    labels = load_labels(batch, args)
    batch, labels = apply_limit(batch, labels, args.limit)
    pipeline = build_pipeline(
        load_settings(),
        args.query_strategy,
        args.query_mode,
    )
    if args.resume_run is None:
        summary = pipeline.run(batch, labels)
    else:
        summary = pipeline.resume(
            args.resume_run,
            batch,
            labels,
            retry_failed=args.retry_failed,
        )
    print_summary(summary)


if __name__ == "__main__":
    main()
