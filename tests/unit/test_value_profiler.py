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


def test_profile_detects_x_masked_mobile_numbers(profiler: ValueProfiler):
    result = profiler.profile("column_x", ["138XX1234"])

    assert result.candidate_types[:2] == ["手机号码", "联系方式"]
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


@pytest.mark.parametrize(
    ("field_name", "samples", "expected_candidate"),
    [
        ("column_x", ["192.168.1.10", "8.8.8.8"], "IPv4地址"),
        ("column_x", ["11010519491231002X"], "身份证号"),
        ("column_x", ["4532015112830366"], "银行卡号"),
        ("column_x", ["2026-09-04", "2026-09-05"], "日期时间"),
        (
            "column_x",
            ["550e8400-e29b-41d4-a716-446655440000"],
            "UUID",
        ),
        ("column_x", ["https://example.com/account?id=1"], "URL"),
        ("column_x", ["31.2304,121.4737"], "经纬度"),
        ("column_x", ["490154203237518"], "IMEI设备编号"),
    ],
)
def test_profile_detects_required_formats(
    profiler: ValueProfiler,
    field_name: str,
    samples: list[str],
    expected_candidate: str,
):
    result = profiler.profile(field_name, samples)

    assert expected_candidate in result.candidate_types


def test_profile_describes_plain_number_without_forcing_type(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["42", "108", "900"])

    assert "普通数字" in result.features
    assert result.candidate_types == []


def test_profile_describes_plain_text_without_forcing_type(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["pending", "approved"])

    assert "普通文本" in result.features
    assert result.candidate_types == []


def test_profile_reports_character_composition_ratios(profiler: ValueProfiler):
    result = profiler.profile("column_x", ["AB12中文"])

    assert "数字字符占比33%" in result.features
    assert "字母字符占比33%" in result.features
    assert "中文字符占比33%" in result.features
    assert "十六进制字符占比67%" in result.features


def test_profile_limits_and_deduplicates_candidates(profiler: ValueProfiler):
    result = profiler.profile("device_contact", ["490154203237518"])

    assert len(result.candidate_types) <= 3
    assert len(result.candidate_types) == len(set(result.candidate_types))


@pytest.mark.parametrize(
    ("sample", "expected_candidate"),
    [
        ("110105********002X", "身份证号"),
        ("6222********1234", "银行卡号"),
    ],
)
def test_profile_detects_masked_numeric_identifiers(
    profiler: ValueProfiler,
    sample: str,
    expected_candidate: str,
):
    result = profiler.profile("column_x", [sample])

    assert expected_candidate in result.candidate_types
    assert "存在脱敏字符" in result.features


def test_profile_does_not_treat_ordinary_x_letters_as_masking(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["boxx_status"])

    assert "存在脱敏字符" not in result.features
