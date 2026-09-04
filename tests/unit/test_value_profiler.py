import pytest

from app.services.value_profiler import ValueProfile, ValueProfiler


@pytest.fixture
def profiler() -> ValueProfiler:
    return ValueProfiler()


def test_profile_detects_masked_mobile_numbers(profiler: ValueProfiler):
    result = profiler.profile("attr_01", ["138**1234", "159**5678"])

    assert isinstance(result, ValueProfile)
    assert result.candidate_types[:2] == ["手机号码", "联系方式"]
    assert "存在脱敏字符" in result.features
    assert "多个样例格式一致" in result.features


def test_profile_detects_mac_from_value_with_meaningless_name(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["A1:B2:C3:D4:E5:F6"])

    assert result.candidate_types[:2] == ["MAC地址", "设备标识信息"]
    assert "6组十六进制字符" in result.features
    assert "冒号分隔" in result.features
    assert "格式稳定" in result.features


def test_profile_detects_masked_email(profiler: ValueProfiler):
    result = profiler.profile("attr_01", ["test***@example.com"])

    assert result.candidate_types[:2] == ["邮箱", "联系方式"]
    assert "包含@符号" in result.features
    assert "存在脱敏字符" in result.features


def test_profile_does_not_force_candidate_for_product_names(
    profiler: ValueProfiler,
):
    result = profiler.profile("product_name", ["无线鼠标", "机械键盘", "显示器"])

    assert result.candidate_types == []
    assert "主要由中文组成" in result.features


def test_profile_returns_empty_candidates_without_samples(profiler: ValueProfiler):
    result = profiler.profile("phone_number", ["", "   "])

    assert result.features == ["无有效样例"]
    assert result.candidate_types == []
