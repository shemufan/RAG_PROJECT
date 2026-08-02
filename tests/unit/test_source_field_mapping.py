from types import SimpleNamespace

import pytest

from app.repositories.source_mysql import (
    SourceMySQLRepository,
    business_domain_for_table,
    map_metadata_row,
    mask_sample_value,
    validate_sample_limit,
)

METADATA_ROW = {
    "TABLE_SCHEMA": "enterprise_source",
    "TABLE_NAME": "employee",
    "TABLE_COMMENT": "员工信息",
    "COLUMN_NAME": "phone",
    "COLUMN_TYPE": "varchar(20)",
    "COLUMN_COMMENT": "联系电话",
    "IS_NULLABLE": "NO",
    "COLUMN_KEY": "",
    "ORDINAL_POSITION": 4,
}


def test_information_schema_row_maps_to_field_profile():
    profile = map_metadata_row(METADATA_ROW, ["13812345678"])

    assert profile.source_system == "mysql"
    assert profile.database_name == "enterprise_source"
    assert profile.table_name == "employee"
    assert profile.table_comment == "员工信息"
    assert profile.field_name == "phone"
    assert profile.field_comment == "联系电话"
    assert profile.data_type == "varchar(20)"
    assert profile.is_nullable is False
    assert profile.column_key is None
    assert profile.business_domain == "hr"
    assert profile.sample_values == ["138****5678"]


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("id_card_no", "110101199001011234", "1101**********1234"),
        ("mobile", "13812345678", "138****5678"),
        ("bank_card_no", "6222021234567890123", "6222***********0123"),
        ("email", "alice@example.test", "a****@example.test"),
        ("description", "x" * 60, "x" * 50),
    ],
)
def test_sample_masking(name, value, expected):
    assert mask_sample_value(name, value) == expected


def test_sample_limit_boundaries():
    assert validate_sample_limit(0) == 0
    assert validate_sample_limit(5) == 5
    with pytest.raises(ValueError):
        validate_sample_limit(-1)
    with pytest.raises(ValueError):
        validate_sample_limit(6)


@pytest.mark.parametrize(
    ("table_name", "domain"),
    [
        ("employee", "hr"),
        ("customer_account", "customer"),
        ("customer_order", "commerce"),
        ("product", "product"),
        ("unknown", "general"),
    ],
)
def test_business_domain_mapping_is_explicit(table_name, domain):
    assert business_domain_for_table(table_name) == domain


def test_mapping_deduplicates_masked_samples_and_drops_nulls():
    profile = map_metadata_row(
        METADATA_ROW,
        [None, "13812345678", "13812345678", "13900001234"],
    )

    assert profile.sample_values == ["138****5678", "139****1234"]


def test_scan_fields_maps_every_metadata_row_without_classification_calls():
    class FakeEngine:
        url = SimpleNamespace(database="enterprise_source")

    class StubSourceRepository(SourceMySQLRepository):
        def _load_metadata(self, table_names):
            assert table_names == ["employee"]
            return [METADATA_ROW]

        def _load_samples(self, row, sample_limit):
            assert row["COLUMN_NAME"] == "phone"
            assert sample_limit == 3
            return ["13812345678"]

    repository = StubSourceRepository(engine=FakeEngine())

    profiles = repository.scan_fields(sample_limit=3, table_names=["employee"])

    assert len(profiles) == 1
    assert profiles[0].field_name == "phone"
    assert profiles[0].sample_values == ["138****5678"]
