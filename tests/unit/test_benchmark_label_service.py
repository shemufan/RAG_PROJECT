import csv

import pytest

from app.services.benchmark_label_service import attach_benchmark_labels
from app.services.csv_reader import CSVInputError
from app.services.tabular_csv_adapter import TabularCSVAdapter


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def make_batch(tmp_path):
    source = tmp_path / "business.csv"
    write_csv(source, [["reg_ip", "price", "created_at"], ["10.*.*.*", "99", "2026"]])
    return TabularCSVAdapter().load(source)


@pytest.mark.parametrize(
    ("raw_true", "raw_false"),
    [("true", "false"), ("TRUE", "FALSE"), ("1", "0")],
)
def test_labels_attach_outside_field_profile(tmp_path, raw_true, raw_false):
    batch = make_batch(tmp_path)
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [["field_name", "expected_personal"], [" reg_ip ", raw_true], ["price", raw_false]],
    )

    summary = attach_benchmark_labels(batch, labels)

    assert summary.labeled_cases == 2
    assert summary.unlabeled_cases == 1
    assert [case.expected_personal for case in summary.cases] == [True, False, None]
    assert "expected_personal" not in summary.cases[0].field_profile.model_dump()
    assert len(summary.label_fingerprint) == 64


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [["field_name", "expected_personal"], ["reg_ip", "true"], ["reg_ip", "false"]],
            "duplicate",
        ),
        (
            [["field_name", "expected_personal"], ["reg_ip", "yes"]],
            "boolean",
        ),
        (
            [["field_name", "expected_personal"], ["not_in_source", "true"]],
            "not present",
        ),
    ],
)
def test_labels_reject_invalid_or_mismatched_rows(tmp_path, rows, message):
    batch = make_batch(tmp_path)
    labels = tmp_path / "labels.csv"
    write_csv(labels, rows)

    with pytest.raises(CSVInputError, match=message):
        attach_benchmark_labels(batch, labels)


def test_labels_require_exact_header(tmp_path):
    batch = make_batch(tmp_path)
    labels = tmp_path / "labels.csv"
    write_csv(labels, [["field", "label"], ["reg_ip", "true"]])

    with pytest.raises(CSVInputError, match="header"):
        attach_benchmark_labels(batch, labels)
