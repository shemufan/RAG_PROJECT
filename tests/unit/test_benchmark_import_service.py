import csv

import pytest

from app.services.benchmark_import_service import (
    BenchmarkImportError,
    parse_benchmark_csv,
)


def write_csv(path, encoding, rows):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["字段名", *[f"样本{index}" for index in range(1, 11)]])
        writer.writerows(rows)


def test_parser_supports_utf8_bom_and_assigns_personal_label(tmp_path):
    path = tmp_path / "personal.csv"
    write_csv(path, "utf-8-sig", [["reg_ip", "192.168.*.*", "10.0.*.*"]])

    rows = parse_benchmark_csv(path, source_dataset="personal")

    assert rows[0].field_name == "reg_ip"
    assert rows[0].sample_values == ["192.168.*.*", "10.0.*.*"]
    assert rows[0].expected_personal is True
    assert rows[0].source_row_number == 2


def test_parser_supports_gb18030_limits_samples_and_text(tmp_path):
    path = tmp_path / "non-personal.csv"
    write_csv(path, "gb18030", [["description", *("x" * 60 for _ in range(7))]])

    rows = parse_benchmark_csv(path, source_dataset="non_personal")

    assert rows[0].expected_personal is False
    assert len(rows[0].sample_values) == 5
    assert all(len(sample) == 50 for sample in rows[0].sample_values)


def test_parser_preserves_duplicate_field_names_on_different_rows(tmp_path):
    path = tmp_path / "personal.csv"
    write_csv(path, "utf-8", [["email", "a***@x.test"], ["email", "b***@x.test"]])

    rows = parse_benchmark_csv(path, source_dataset="personal")

    assert [row.source_row_number for row in rows] == [2, 3]
    assert [row.field_name for row in rows] == ["email", "email"]


def test_parser_rejects_missing_field_name_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,样本1\nemail,a***@x.test\n", encoding="utf-8")

    with pytest.raises(BenchmarkImportError, match="字段名"):
        parse_benchmark_csv(path, source_dataset="personal")
