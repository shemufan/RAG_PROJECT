import re
from pathlib import Path

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def normalized(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.replace("`", "").lower())


def test_source_schema_has_required_business_tables_and_no_answers():
    sql = normalized(read_sql("source_schema.sql"))

    for table in ("employee", "customer_account", "customer_order", "product"):
        assert f"create table if not exists {table}" in sql
    for forbidden in ("sensitivity", "classification_level", "category_level"):
        assert forbidden not in sql
    assert sql.count("comment=") == 4
    assert "employee_id varchar" in sql
    assert "bank_card_no varchar" in sql
    assert "transaction_no varchar" in sql


def test_source_seed_inserts_five_rows_per_table_without_answers():
    sql = read_sql("source_seed.sql")

    for table in ("employee", "customer_account", "customer_order", "product"):
        match = re.search(
            rf"INSERT INTO `{table}`.*?VALUES\s*(.*?);",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert match is not None
        assert len(re.findall(r"^\s*\(", match.group(1), flags=re.MULTILINE)) == 5
    assert "@example.test" in sql
    assert "classification" not in sql.lower()


def test_target_schema_has_required_tables_constraints_and_indexes():
    sql = normalized(read_sql("target_schema.sql"))

    for table in (
        "classification_run",
        "data_field_asset",
        "field_classification_result",
        "classification_evidence",
    ):
        assert f"create table if not exists {table}" in sql
    assert "unique key uq_field_identity" in sql
    assert "unique key uq_run_field" in sql
    assert "on delete cascade" in sql
    assert "is_personal boolean not null" in sql
    for index_name in (
        "idx_run_status",
        "idx_run_started_at",
        "idx_asset_database_table",
        "idx_asset_business_domain",
        "idx_result_level",
        "idx_result_category",
        "idx_result_need_review",
        "idx_result_field_id",
        "idx_evidence_result_id",
        "idx_evidence_document_name",
    ):
        assert index_name in sql


def test_benchmark_schema_files_define_source_and_target_tables():
    source = normalized(read_sql("benchmark_source_schema.sql"))
    target = normalized(read_sql("benchmark_target_schema.sql"))

    assert "create table if not exists benchmark_field_input" in source
    assert "unique key uq_benchmark_source_row" in source
    assert "create table if not exists benchmark_run" in target
    assert "create table if not exists benchmark_prediction" in target
    assert "unique key uq_benchmark_run_case" in target


def test_query_examples_cover_ten_documented_queries():
    sql = read_sql("query_examples.sql")

    assert len(re.findall(r"^-- Query \d+:", sql, flags=re.MULTILINE)) == 10
    assert "r.confidence < 0.75" in sql
    assert "r.need_review = TRUE" in sql
    assert "r.level IN ('L3', 'L4')" in sql
    assert "ORDER BY cr.started_at DESC" in sql
