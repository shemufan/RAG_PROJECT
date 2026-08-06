import csv

import pytest

from scripts.run_csv_pipeline import load_csv_batch, parse_args


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
