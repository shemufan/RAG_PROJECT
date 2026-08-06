import pytest

from app.services.csv_mode_detector import CSVModeError, resolve_csv_mode


@pytest.mark.parametrize(
    "headers",
    [["字段名", "样本1"], ["field_name", "sample_1"], ["field_name", "sample2"]],
)
def test_auto_detects_strong_catalog_signature(headers):
    assert resolve_csv_mode(headers, "auto") == "catalog"


def test_auto_defaults_ordinary_business_headers_to_tabular():
    assert resolve_csv_mode(["user_id", "reg_ip", "created_at"], "auto") == "tabular"


def test_auto_rejects_ambiguous_unknown_catalog_metadata():
    with pytest.raises(CSVModeError, match="ambiguous"):
        resolve_csv_mode(["字段名", "样本1", "未知元数据"], "auto")


def test_explicit_mode_overrides_detection():
    assert resolve_csv_mode(["字段名", "样本1"], "tabular") == "tabular"
    assert resolve_csv_mode(["only_name"], "catalog") == "catalog"
