import csv

import pytest

from app.services.catalog_csv_adapter import CatalogCSVAdapter
from app.services.csv_reader import CSVInputError


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_catalog_maps_rows_aliases_and_numeric_sample_order(tmp_path):
    path = tmp_path / "personal-labeled-name.csv"
    write_csv(
        path,
        [
            ["字段名", "字段说明", "数据类型", "样本10", "样本2", "样本1"],
            ["reg_ip", "注册IP", "varchar", "ten", "two", "one"],
            ["reg_ip", "另一条记录", "varchar", "", "second", "first"],
        ],
    )

    batch = CatalogCSVAdapter().load(path)

    assert batch.input_mode == "catalog"
    assert [case.field_profile.field_name for case in batch.cases] == ["reg_ip", "reg_ip"]
    assert batch.cases[0].field_profile.sample_values == ["one", "two", "ten"]
    assert batch.cases[0].field_profile.field_comment == "注册IP"
    assert batch.cases[0].field_profile.database_name == "csv_source"
    assert "personal" not in batch.cases[0].field_profile.model_dump_json()


def test_catalog_supports_explicit_columns_and_no_samples(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名称", "示例甲"], ["email", "a***@x.test"]])

    mapped = CatalogCSVAdapter().load(
        path,
        field_name_column="字段名称",
        sample_columns=["示例甲"],
    )
    no_samples = CatalogCSVAdapter().load(
        path,
        field_name_column="字段名称",
        sample_columns=[],
    )

    assert mapped.cases[0].field_profile.sample_values == ["a***@x.test"]
    assert no_samples.cases[0].field_profile.sample_values == []


def test_catalog_requires_a_field_name_column(tmp_path):
    path = tmp_path / "invalid.csv"
    write_csv(path, [["other", "样本1"], ["x", "value"]])

    with pytest.raises(CSVInputError, match="field-name column"):
        CatalogCSVAdapter().load(path)
