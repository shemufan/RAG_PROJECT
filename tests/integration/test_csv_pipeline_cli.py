import csv

import pytest

from scripts.run_csv_pipeline import apply_limit, load_csv_batch, load_labels, parse_args


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_cli_auto_loads_tabular_and_catalog_inputs(tmp_path):
    tabular = tmp_path / "business.csv"
    catalog = tmp_path / "catalog.csv"
    write_csv(tabular, [["email", "price"], ["a***@x.test", "99"]])
    write_csv(catalog, [["字段名", "样本1"], ["email", "a***@x.test"]])

    tabular_batch = load_csv_batch(parse_args(["--input", str(tabular)]))
    catalog_batch = load_csv_batch(parse_args(["--input", str(catalog)]))

    assert tabular_batch.input_mode == "tabular"
    assert catalog_batch.input_mode == "catalog"


def test_cli_accepts_resume_and_rejects_invalid_combinations(tmp_path):
    path = tmp_path / "business.csv"
    write_csv(path, [["email"], ["a***@x.test"]])
    run_id = "12345678-1234-5678-1234-567812345678"

    resumed = parse_args(
        ["--resume-run", run_id, "--input", str(path), "--retry-failed"]
    )
    assert str(resumed.resume_run) == run_id

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--retry-failed"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--input-mode",
                "tabular",
                "--field-name-column",
                "name",
            ]
        )


def test_cli_loads_embedded_label_column(tmp_path):
    source = tmp_path / "catalog.csv"
    write_csv(
        source,
        [
            ["字段名", "样本1", "expected_personal"],
            ["email", "a***@x.test", "true"],
            ["price", "99", "false"],
        ],
    )

    args = parse_args(["--input", str(source), "--label-column", "expected_personal"])
    batch = load_csv_batch(args)
    labels = load_labels(batch, args)

    assert batch.input_mode == "catalog"
    assert [case.expected_personal for case in labels.cases] == [True, False]


def test_cli_rejects_label_column_with_separate_labels(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1", "expected_personal"], ["email", "a", "true"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--label-column",
                "expected_personal",
                "--labels",
                "labels.csv",
            ]
        )


def test_cli_rejects_label_column_with_tabular_mode(tmp_path):
    path = tmp_path / "business.csv"
    write_csv(path, [["email", "price"], ["a", "1"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--label-column",
                "expected_personal",
                "--input-mode",
                "tabular",
            ]
        )


def test_cli_limit_truncates_batch_and_labels(tmp_path):
    source = tmp_path / "catalog.csv"
    write_csv(
        source,
        [
            ["字段名", "样本1", "expected_personal"],
            ["email", "a", "true"],
            ["price", "99", "false"],
            ["ip", "1.2.3.4", "true"],
        ],
    )

    args = parse_args(
        ["--input", str(source), "--label-column", "expected_personal", "--limit", "2"]
    )
    batch = load_csv_batch(args)
    labels = load_labels(batch, args)
    limited_batch, limited_labels = apply_limit(batch, labels, args.limit)

    assert [case.field_profile.field_name for case in limited_batch.cases] == [
        "email",
        "price",
    ]
    assert [case.expected_personal for case in limited_labels.cases] == [True, False]
    assert limited_labels.labeled_cases == 2
    assert limited_labels.unlabeled_cases == 0


def test_cli_rejects_limit_with_resume(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a"]])
    run_id = "12345678-1234-5678-1234-567812345678"

    with pytest.raises(SystemExit):
        parse_args(
            ["--input", str(path), "--limit", "5", "--resume-run", run_id]
        )
