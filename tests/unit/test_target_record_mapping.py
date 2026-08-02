from uuid import uuid4

import pytest
from pydantic import ValidationError

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


def test_stable_field_id_is_repeatable_and_field_specific():
    profile = make_profile()

    assert stable_field_id(profile) == stable_field_id(profile)
    assert stable_field_id(profile) != stable_field_id(
        profile.model_copy(update={"field_name": "phone"})
    )


def test_field_classification_record_keeps_input_output_and_evidence():
    profile = make_profile()
    record = FieldClassificationRecord(
        run_id=uuid4(),
        field_id=stable_field_id(profile),
        field_profile=profile,
        classification=ClassificationOutput(
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
            )
        ],
        decision_path="rag_llm",
        model_name="deepseek-chat",
        knowledge_base_version="v1",
    )

    assert record.field_id == stable_field_id(profile)
    assert record.evidence[0].source == "个人信息保护法.txt"
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
