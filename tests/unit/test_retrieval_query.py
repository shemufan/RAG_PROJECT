import subprocess
import sys

import pytest

from app.rag.retrieval_query import (
    CleanQueryBuilder,
    LegacyQueryBuilder,
    RetrievalQueryBuilder,
    create_query_builder,
)
from app.schemas.field import FieldProfile
from app.services.value_profiler import ValueProfile


def noisy_field() -> FieldProfile:
    return FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="device_attr",
        field_cn="设备属性",
        field_comment="采集终端设备属性",
        data_type="unknown",
        sample_values=["A1:B2:C3:D4:E5:F6", "11:22:33:44:55:66"],
        business_domain="general",
    )


class StaticProfiler:
    def __init__(self, value_profile: ValueProfile):
        self.value_profile = value_profile
        self.call_count = 0

    def profile(self, field_name: str, sample_values: list[str]) -> ValueProfile:
        self.call_count += 1
        return self.value_profile


def test_legacy_builder_matches_original_query_exactly():
    assert LegacyQueryBuilder().build(noisy_field()) == (
        "field_name: device_attr\n"
        "field_cn: 设备属性\n"
        "field_comment: 采集终端设备属性\n"
        "data_type: unknown\n"
        "sample_values: A1:B2:C3:D4:E5:F6、11:22:33:44:55:66\n"
        "business_domain: general\n"
        "table_name: catalog_input\n"
        "database_name: csv_source\n"
        "source_system: csv"
    )


def test_clean_builder_contains_only_original_core_values():
    query = CleanQueryBuilder().build(noisy_field())

    assert query == (
        "field_name: device_attr\n"
        "sample_values: A1:B2:C3:D4:E5:F6、11:22:33:44:55:66"
    )
    for forbidden in (
        "设备属性",
        "采集终端设备属性",
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
    ):
        assert forbidden not in query


def test_profile_builder_calls_value_profiler():
    class StubProfiler:
        def profile(self, field_name: str, sample_values: list[str]):
            assert field_name == "device_attr"
            assert sample_values == [
                "A1:B2:C3:D4:E5:F6",
                "11:22:33:44:55:66",
            ]
            return ValueProfile(
                features=["6组十六进制字符", "冒号分隔"],
                candidate_types=["MAC地址", "设备标识信息"],
            )

    query = RetrievalQueryBuilder(value_profiler=StubProfiler()).build(noisy_field())

    assert "字段名：device_attr" in query
    assert "数据结构特征：6组十六进制字符、冒号分隔" in query
    assert "候选数据类型：MAC地址、设备标识信息" in query


@pytest.mark.parametrize(
    ("mode", "has_features", "has_candidates"),
    [
        ("c", True, True),
        ("c1", True, False),
        ("c2", False, True),
    ],
)
def test_profile_submode_controls_only_enriched_sections(
    mode: str,
    has_features: bool,
    has_candidates: bool,
):
    profiler = StaticProfiler(
        ValueProfile(
            features=["6组十六进制字符", "冒号分隔"],
            candidate_types=["MAC地址", "设备标识信息"],
        )
    )
    builder = RetrievalQueryBuilder(
        value_profiler=profiler,
        profile_mode=mode,
    )

    query = builder.build(noisy_field())

    assert "字段名：device_attr" in query
    assert "样例值：A1:B2:C3:D4:E5:F6、11:22:33:44:55:66" in query
    assert ("数据结构特征：" in query) is has_features
    assert ("候选数据类型：" in query) is has_candidates
    assert profiler.call_count == 1


def test_profile_builder_rejects_unknown_submode():
    with pytest.raises(ValueError, match="unsupported profile query mode"):
        RetrievalQueryBuilder(profile_mode="other")


@pytest.mark.parametrize(
    ("strategy", "builder_type"),
    [
        ("legacy", LegacyQueryBuilder),
        ("clean", CleanQueryBuilder),
        ("profile", RetrievalQueryBuilder),
    ],
)
def test_factory_returns_requested_builder(strategy: str, builder_type: type):
    assert isinstance(create_query_builder(strategy), builder_type)


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unsupported query strategy"):
        create_query_builder("other")


def test_factory_passes_profile_submode_to_profile_builder():
    query = create_query_builder("profile", profile_mode="c1").build(noisy_field())

    assert "数据结构特征：" in query
    assert "候选数据类型：" not in query


def test_factory_injects_selected_value_profiler():
    profiler = StaticProfiler(
        ValueProfile(
            features=["LLM结构特征"],
            candidate_types=["LLM候选类型"],
        )
    )
    builder = create_query_builder(
        "profile",
        profile_mode="c2",
        value_profiler=profiler,
    )

    query = builder.build(noisy_field())

    assert profiler.call_count == 1
    assert "候选数据类型：LLM候选类型" in query
    assert "数据结构特征：" not in query


def test_profile_builder_selects_at_most_three_representative_values():
    builder = RetrievalQueryBuilder(
        value_profiler=StaticProfiler(ValueProfile(features=["存在脱敏字符"]))
    )
    query = builder.build(
        FieldProfile(
            field_name="contact",
            sample_values=[
                "138**1234",
                "138**1234",
                "159**5678",
                "test***@example.com",
                "010-12345678",
            ],
        )
    )

    assert "138**1234、test***@example.com、010-12345678" in query
    assert "159**5678" not in query


def test_profile_builder_omits_empty_optional_sections():
    builder = RetrievalQueryBuilder(value_profiler=StaticProfiler(ValueProfile()))

    query = builder.build(FieldProfile(field_name="product_name"))

    assert query == "字段名：product_name"
    assert "样例值" not in query
    assert "候选数据类型" not in query


def test_profile_builder_does_not_emit_field_profile_noise():
    builder = RetrievalQueryBuilder(
        value_profiler=StaticProfiler(
            ValueProfile(
                features=["6组十六进制字符"],
                candidate_types=["MAC地址"],
            )
        )
    )

    query = builder.build(noisy_field())

    for forbidden in (
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
        "field_cn",
        "field_comment",
        "database_name",
        "table_name",
        "source_system",
    ):
        assert forbidden not in query


def test_retrieval_query_module_can_be_imported_first_in_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.rag.retrieval_query import RetrievalQueryBuilder",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
