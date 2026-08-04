import pytest
from pydantic import ValidationError

from app.schemas.classification import (
    ClassificationOutput,
    ClassificationResult,
    Evidence,
)
from app.schemas.field import FieldProfile


def test_field_profile_validates_required_name_and_uses_independent_samples():
    first = FieldProfile(field_name=" user_id ")
    second = FieldProfile(field_name="account_id")

    first.sample_values.append("42")

    assert first.field_name == "user_id"
    assert second.sample_values == []
    with pytest.raises(ValidationError):
        FieldProfile(field_name="   ")


def test_field_profile_rejects_oversized_input():
    with pytest.raises(ValidationError):
        FieldProfile(field_name="x" * 129)
    with pytest.raises(ValidationError):
        FieldProfile(field_name="name", sample_values=["1", "2", "3", "4", "5", "6"])
    with pytest.raises(ValidationError):
        FieldProfile(field_name="name", sample_values=["x" * 51])


def test_field_profile_has_pipeline_metadata_and_manual_defaults():
    profile = FieldProfile(field_name="employee_id")

    assert profile.schema_version == "1.0"
    assert profile.source_system == "manual"
    assert profile.database_name == "manual"
    assert profile.table_name == "manual"
    assert profile.data_type == "unknown"
    assert profile.is_nullable is True


@pytest.mark.parametrize("field", ["field_name", "database_name", "table_name"])
def test_field_profile_rejects_empty_physical_identity(field):
    payload = {"field_name": "name", field: ""}

    with pytest.raises(ValidationError):
        FieldProfile(**payload)


def test_classification_output_rejects_invalid_level_and_confidence():
    with pytest.raises(ValidationError):
        ClassificationOutput(
            is_personal=True,
            category="个人信息",
            level="L5",
            confidence=1.1,
            reason="invalid",
            need_review=False,
        )


def test_classification_result_keeps_retrieval_evidence():
    evidence = Evidence(
        content="身份证件号码属于敏感个人信息。",
        source="个人信息保护法.txt",
        article="第二十八条",
        score=0.91,
        chunk_id="chunk-1",
    )

    result = ClassificationResult(
        field_name="id_card",
        is_personal=True,
        category="敏感个人信息",
        subcategory="身份标识",
        level="L4",
        confidence=0.92,
        reason="法规明确规定。",
        evidence=[evidence],
        need_review=False,
        decision_path="rag_llm",
    )

    assert result.evidence[0].chunk_id == "chunk-1"
    assert result.is_personal is True
