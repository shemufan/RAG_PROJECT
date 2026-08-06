"""Attach optional ground-truth labels without changing model input profiles."""

from pathlib import Path

from app.schemas.csv_input import CSVInputBatch, LabelMatchSummary
from app.services.csv_reader import CSVInputError, CSVReader

LABEL_HEADERS = ["field_name", "expected_personal"]
TRUE_VALUES = {"true", "1"}
FALSE_VALUES = {"false", "0"}


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
