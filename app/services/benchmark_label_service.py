"""Attach optional ground-truth labels without changing model input profiles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.schemas.csv_input import (
    CSVInputBatch,
    CSVProfileDefaults,
    LabelMatchSummary,
)
from app.services.catalog_csv_adapter import CatalogCSVAdapter
from app.services.csv_reader import CSVInputError, CSVReader

LABEL_HEADERS = ["field_name", "expected_personal"]
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}
BENCHMARK_INPUT_VERSION = "benchmark-catalog-v1"
BENCHMARK_PROFILE_DEFAULTS = CSVProfileDefaults(
    source_system="benchmark",
    database_name="teacher_benchmark",
    table_name="benchmark_input",
    data_type="unknown",
    business_domain="general",
)


@dataclass(frozen=True)
class BenchmarkCSVInput:
    batch: CSVInputBatch
    labels: LabelMatchSummary


def _parse_boolean(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise CSVInputError(f"label row {row_number} has an invalid boolean")


def attach_benchmark_labels(
    batch: CSVInputBatch,
    label_path: str | Path,
    *,
    reader: CSVReader | None = None,
) -> LabelMatchSummary:
    csv_reader = reader or CSVReader()
    inspection = csv_reader.inspect(label_path)
    if inspection.headers != LABEL_HEADERS:
        raise CSVInputError("label CSV header must be field_name,expected_personal")

    labels: dict[str, bool] = {}
    for row_number, row in csv_reader.iter_rows(label_path, inspection):
        field_name = row["field_name"].strip()
        if not field_name:
            raise CSVInputError(f"label row {row_number} has an empty field name")
        if field_name in labels:
            raise CSVInputError(f"duplicate label for field {field_name}")
        labels[field_name] = _parse_boolean(row["expected_personal"], row_number)

    source_fields = {case.field_profile.field_name for case in batch.cases}
    extra_labels = labels.keys() - source_fields
    if extra_labels:
        raise CSVInputError("label field is not present in the input CSV")

    cases = [
        case.model_copy(
            update={"expected_personal": labels.get(case.field_profile.field_name)}
        )
        for case in batch.cases
    ]
    labeled_cases = sum(case.expected_personal is not None for case in cases)
    return LabelMatchSummary(
        label_fingerprint=inspection.source_fingerprint,
        labeled_cases=labeled_cases,
        unlabeled_cases=len(cases) - labeled_cases,
        cases=cases,
    )


def _fingerprint(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1")


def prepare_labeled_catalog_benchmark(
    personal_path: str | Path,
    non_personal_path: str | Path,
    *,
    batch_name: str,
    personal_limit: int | None = None,
    non_personal_limit: int | None = None,
    reader: CSVReader | None = None,
) -> BenchmarkCSVInput:
    """Build one labeled Benchmark input using the generic catalog adapter."""
    _validate_limit("personal_limit", personal_limit)
    _validate_limit("non_personal_limit", non_personal_limit)
    csv_reader = reader or CSVReader()
    adapter = CatalogCSVAdapter(csv_reader)
    personal = adapter.load(
        personal_path,
        profile_defaults=BENCHMARK_PROFILE_DEFAULTS,
    )
    non_personal = adapter.load(
        non_personal_path,
        profile_defaults=BENCHMARK_PROFILE_DEFAULTS,
    )
    personal_cases = personal.cases[:personal_limit]
    non_personal_cases = non_personal.cases[:non_personal_limit]
    combined = [*personal_cases, *non_personal_cases]
    cases = [
        case.model_copy(update={"case_index": index, "expected_personal": None})
        for index, case in enumerate(combined, start=1)
    ]
    labeled_cases = [
        case.model_copy(update={"expected_personal": index <= len(personal_cases)})
        for index, case in enumerate(cases, start=1)
    ]
    fingerprint_payload = {
        "version": BENCHMARK_INPUT_VERSION,
        "personal": personal.source_fingerprint,
        "non_personal": non_personal.source_fingerprint,
        "personal_limit": personal_limit,
        "non_personal_limit": non_personal_limit,
        "profile_defaults": BENCHMARK_PROFILE_DEFAULTS.model_dump(),
    }
    source_fingerprint = _fingerprint(
        {"purpose": "benchmark-source", **fingerprint_payload}
    )
    label_fingerprint = _fingerprint(
        {"purpose": "benchmark-labels", **fingerprint_payload}
    )
    return BenchmarkCSVInput(
        batch=CSVInputBatch(
            source_name=batch_name,
            source_fingerprint=source_fingerprint,
            input_mode="catalog",
            personal_limit=personal_limit,
            non_personal_limit=non_personal_limit,
            cases=cases,
        ),
        labels=LabelMatchSummary(
            label_fingerprint=label_fingerprint,
            labeled_cases=len(labeled_cases),
            unlabeled_cases=0,
            cases=labeled_cases,
        ),
    )
