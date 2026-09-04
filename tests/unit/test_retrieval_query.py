import subprocess
import sys

from app.rag.retrieval_query import RetrievalQueryBuilder
from app.services.value_profiler import ValueProfile


def test_builder_serializes_only_clean_retrieval_signals():
    query = RetrievalQueryBuilder().build(
        "device_attr",
        ["A1:B2:C3:D4:E5:F6"],
        ValueProfile(
            features=["6组十六进制字符", "冒号分隔", "格式稳定"],
            candidate_types=["MAC地址", "设备标识信息"],
        ),
    )

    assert query == (
        "字段名：device_attr "
        "样例值：A1:B2:C3:D4:E5:F6 "
        "数据结构特征：6组十六进制字符、冒号分隔、格式稳定 "
        "候选数据类型：MAC地址、设备标识信息"
    )


def test_builder_selects_at_most_three_distinct_representative_values():
    query = RetrievalQueryBuilder().build(
        "contact",
        [
            "138**1234",
            "138**1234",
            "159**5678",
            "test***@example.com",
            "010-12345678",
        ],
        ValueProfile(features=["存在脱敏字符"]),
    )

    assert "138**1234、test***@example.com、010-12345678" in query
    assert "159**5678" not in query


def test_builder_omits_empty_optional_sections():
    query = RetrievalQueryBuilder().build(
        "product_name",
        [],
        ValueProfile(features=[], candidate_types=[]),
    )

    assert query == "字段名：product_name"
    assert "样例值" not in query
    assert "候选数据类型" not in query


def test_builder_cannot_receive_or_emit_field_profile_noise():
    query = RetrievalQueryBuilder().build(
        "attr_01",
        ["A1:B2:C3:D4:E5:F6"],
        ValueProfile(
            features=["6组十六进制字符"],
            candidate_types=["MAC地址"],
        ),
    )

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
