import csv

import pytest

from app.services.benchmark_label_service import (
    attach_benchmark_labels,
    prepare_labeled_catalog_benchmark,
)
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


def test_two_file_benchmark_uses_generic_catalog_profiles_and_role_labels(tmp_path):
    personal = tmp_path / "personal.csv"
    non_personal = tmp_path / "non_personal.csv"
    write_csv(
        personal,
        [
            ["field_name", "source_system", "sample2", "sample1"],
            ["email", "csv-value", "second", "first"],
            ["email", "csv-value", "duplicate", "kept"],
        ],
    )
    write_csv(
        non_personal,
        [["field_name", "sample1"], ["price", "99"], ["created_at", "2026"]],
    )

    prepared = prepare_labeled_catalog_benchmark(
        personal,
        non_personal,
        batch_name="teacher_2026_08",
    )

    assert prepared.batch.source_name == "teacher_2026_08"
    assert prepared.batch.input_mode == "catalog"
    assert [case.case_index for case in prepared.batch.cases] == [1, 2, 3, 4]
    assert [case.field_profile.field_name for case in prepared.batch.cases] == [
        "email",
        "email",
        "price",
        "created_at",
    ]
    assert [case.expected_personal for case in prepared.batch.cases] == [None] * 4
    assert [case.expected_personal for case in prepared.labels.cases] == [
        True,
        True,
        False,
        False,
    ]
    profile = prepared.batch.cases[0].field_profile
    assert profile.sample_values == ["first", "second"]
    assert profile.source_system == "benchmark"
    assert profile.database_name == "teacher_benchmark"
    assert profile.table_name == "benchmark_input"
    assert profile.data_type == "unknown"
    assert profile.business_domain == "general"
    assert "expected_personal" not in profile.model_dump()
    assert prepared.labels.labeled_cases == 4
    assert prepared.labels.unlabeled_cases == 0
    assert len(prepared.batch.source_fingerprint) == 64
    assert len(prepared.labels.label_fingerprint) == 64


def test_two_file_benchmark_applies_independent_limits_and_fingerprints_inputs(tmp_path):
    personal = tmp_path / "personal.csv"
    non_personal = tmp_path / "non_personal.csv"
    write_csv(personal, [["field_name", "sample1"], ["email", "a"], ["phone", "b"]])
    write_csv(
        non_personal,
        [["field_name", "sample1"], ["price", "1"], ["created_at", "2"]],
    )

    limited = prepare_labeled_catalog_benchmark(
        personal,
        non_personal,
        batch_name="teacher",
        personal_limit=1,
        non_personal_limit=1,
    )
    full = prepare_labeled_catalog_benchmark(
        personal,
        non_personal,
        batch_name="teacher",
    )
    swapped = prepare_labeled_catalog_benchmark(
        non_personal,
        personal,
        batch_name="teacher",
        personal_limit=1,
        non_personal_limit=1,
    )

    assert [case.field_profile.field_name for case in limited.batch.cases] == [
        "email",
        "price",
    ]
    assert limited.batch.source_fingerprint != full.batch.source_fingerprint
    assert limited.batch.source_fingerprint != swapped.batch.source_fingerprint
    assert limited.labels.label_fingerprint != swapped.labels.label_fingerprint
