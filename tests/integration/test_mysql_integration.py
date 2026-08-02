import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from app.repositories.source_mysql import SourceMySQLRepository
from app.repositories.target_mysql import TargetMySQLRepository
from app.schemas.classification import ClassificationOutput, Evidence
from app.schemas.pipeline import (
    FieldClassificationRecord,
    PipelineSummary,
    stable_field_id,
)

SOURCE_URL = os.environ.get("MYSQL_TEST_SOURCE_URL")
TARGET_URL = os.environ.get("MYSQL_TEST_TARGET_URL")
pytestmark = pytest.mark.skipif(
    not SOURCE_URL or not TARGET_URL,
    reason="MYSQL_TEST_SOURCE_URL and MYSQL_TEST_TARGET_URL are not configured",
)

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def _test_database_name(database_url: str) -> str:
    database_name = make_url(database_url).database
    if not database_name or "test" not in database_name.lower():
        raise ValueError("MySQL integration URLs must use dedicated test databases")
    return database_name


def _server_engine(database_url: str):
    return create_engine(make_url(database_url).set(database=None), pool_pre_ping=True)


def _drop_database(database_url: str, database_name: str) -> None:
    engine = _server_engine(database_url)
    quoted_name = engine.dialect.identifier_preparer.quote(database_name)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"DROP DATABASE IF EXISTS {quoted_name}")
    finally:
        engine.dispose()


def _execute_sql_file(database_url: str, file_name: str, replacements: dict[str, str]) -> None:
    sql = (SQL_DIR / file_name).read_text(encoding="utf-8")
    for original, replacement in replacements.items():
        sql = sql.replace(original, replacement)
    statements = [statement.strip() for statement in sql.split(";") if statement.strip()]
    engine = _server_engine(database_url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def test_real_mysql_scan_persist_and_query_cycle():
    source_name = _test_database_name(SOURCE_URL)
    target_name = _test_database_name(TARGET_URL)
    source_replacements = {"enterprise_source": source_name}
    target_replacements = {"compliance_result": target_name}
    _drop_database(SOURCE_URL, source_name)
    _drop_database(TARGET_URL, target_name)
    try:
        _execute_sql_file(SOURCE_URL, "source_schema.sql", source_replacements)
        _execute_sql_file(SOURCE_URL, "source_seed.sql", source_replacements)
        _execute_sql_file(TARGET_URL, "target_schema.sql", target_replacements)

        source = SourceMySQLRepository(SOURCE_URL)
        target = TargetMySQLRepository(TARGET_URL)
        fields = source.scan_fields(sample_limit=2)
        assert {field.table_name for field in fields} == {
            "employee",
            "customer_account",
            "customer_order",
            "product",
        }
        profile = next(field for field in fields if field.field_name == "id_card_no")
        now = datetime.now(timezone.utc)
        run_id = uuid4()
        running = PipelineSummary(
            run_id=run_id,
            source_database=source_name,
            total_fields=0,
            success_fields=0,
            review_fields=0,
            failed_fields=0,
            status="RUNNING",
            started_at=now,
        )
        target.create_run(running, "integration-fake", "v-test")
        field_id = stable_field_id(profile)
        target.upsert_field_asset(field_id, profile, now)
        record = FieldClassificationRecord(
            run_id=run_id,
            field_id=field_id,
            field_profile=profile,
            classification=ClassificationOutput(
                category="个人信息",
                subcategory="身份标识",
                level="L4",
                confidence=0.95,
                reason="集成测试的确定性结果",
                need_review=False,
            ),
            evidence=[
                Evidence(
                    source="个人信息保护法.txt",
                    article="第二十八条",
                    content="敏感个人信息包括特定身份信息。",
                    score=0.9,
                    chunk_id="integration-chunk",
                )
            ],
            decision_path="integration_fake",
            model_name="integration-fake",
            knowledge_base_version="v-test",
            created_at=now,
        )
        result_id = target.save_classification_record(record)
        finished = running.model_copy(
            update={
                "total_fields": 1,
                "success_fields": 1,
                "status": "SUCCESS",
                "finished_at": now,
            }
        )
        target.update_run(finished)

        assert target.get_run(run_id).status == "SUCCESS"
        results = target.query_results(run_id=run_id)
        assert results[0].column_name == "id_card_no"
        evidence = target.get_result_evidence(result_id)
        assert evidence[0].document_name == "个人信息保护法.txt"
        assert evidence[0].rank_no == 1
    finally:
        _drop_database(SOURCE_URL, source_name)
        _drop_database(TARGET_URL, target_name)
