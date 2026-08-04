import json
from contextlib import nullcontext
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.repositories.target_mysql import TargetMySQLRepository
from app.schemas.classification import ClassificationOutput, Evidence
from app.schemas.field import FieldProfile
from app.schemas.pipeline import (
    FieldClassificationRecord,
    PipelineRequest,
    stable_field_id,
)


def make_profile() -> FieldProfile:
    return FieldProfile(
        source_system="mysql",
        database_name="enterprise_source",
        table_name="employee",
        table_comment="员工信息",
        field_name="id_card_no",
        field_cn="身份证号码",
        field_comment="员工身份证号码",
        data_type="varchar(18)",
        is_nullable=False,
        business_domain="hr",
    )


def make_record() -> FieldClassificationRecord:
    profile = make_profile()
    return FieldClassificationRecord(
        run_id=uuid4(),
        field_id=stable_field_id(profile),
        field_profile=profile,
        classification=ClassificationOutput(
            is_personal=True,
            category="个人信息",
            subcategory="身份标识",
            level="L4",
            confidence=0.95,
            reason="法规明确规定",
            need_review=False,
        ),
        evidence=[
            Evidence(
                source="个人信息保护法.txt",
                article="第二十八条",
                content="法规内容",
                score=0.9,
                chunk_id="chunk-28",
            ),
            Evidence(
                source="数据安全法.txt",
                article="第二十一条",
                content="分类分级保护要求",
                score=0.8,
                chunk_id="chunk-21",
            ),
        ],
        decision_path="rag_llm",
        model_name="deepseek-chat",
        knowledge_base_version="v1",
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )


def test_stable_field_id_is_repeatable_and_field_specific():
    profile = make_profile()

    assert stable_field_id(profile) == stable_field_id(profile)
    assert stable_field_id(profile) != stable_field_id(
        profile.model_copy(update={"field_name": "phone"})
    )


def test_field_classification_record_keeps_input_output_and_evidence():
    profile = make_profile()
    record = make_record()

    assert record.field_id == stable_field_id(profile)
    assert [item.source for item in record.evidence] == [
        "个人信息保护法.txt",
        "数据安全法.txt",
    ]
    assert record.created_at.tzinfo is not None


@pytest.mark.parametrize("sample_limit", [-1, 6])
def test_pipeline_request_rejects_invalid_sample_limit(sample_limit):
    with pytest.raises(ValidationError):
        PipelineRequest(sample_limit=sample_limit)


def test_pipeline_request_uses_safe_defaults():
    request = PipelineRequest()

    assert request.sample_limit == 3
    assert request.table_names is None
    assert request.continue_on_error is True


class FakeResult:
    lastrowid = 41


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, parameters=None):
        self.calls.append((str(statement), parameters))
        return FakeResult()


class RecordingEngine:
    def __init__(self):
        self.connection = RecordingConnection()

    def begin(self):
        return nullcontext(self.connection)


def parameters_for(connection, sql_fragment):
    for statement, parameters in connection.calls:
        if sql_fragment in statement:
            return parameters
    raise AssertionError(f"SQL fragment not executed: {sql_fragment}")


def test_target_repository_maps_asset_and_record_to_relational_columns():
    engine = RecordingEngine()
    repository = TargetMySQLRepository(engine=engine)
    record = make_record()

    repository.upsert_field_asset(record.field_id, record.field_profile, record.created_at)
    result_id = repository.save_classification_record(record)

    assert result_id == 41
    asset = parameters_for(engine.connection, "INSERT INTO data_field_asset")
    assert asset["column_name"] == "id_card_no"
    assert asset["database_name"] == "enterprise_source"
    result = parameters_for(engine.connection, "INSERT INTO field_classification_result")
    assert result["level"] == "L4"
    assert json.loads(result["input_snapshot_json"])["schema_version"] == "1.0"
    evidence = parameters_for(engine.connection, "INSERT INTO classification_evidence")
    assert evidence[0]["rank_no"] == 1
    assert evidence[1]["rank_no"] == 2
    assert evidence[0]["document_name"] == "个人信息保护法.txt"
    assert evidence[0]["chunk_id"] == "chunk-28"
