from app.rag.semantic_bridge import (
    build_objective_profile,
    build_regulation_bridge_query,
    build_semantic_query,
)
from app.schemas.field import FieldProfile
from app.schemas.semantic import SemanticCard


class ObjectiveOnlyProfiler:
    def profile(self, *_args):
        raise AssertionError("semantic candidate profiling is forbidden")

    def basic_statistics(self, values):
        assert values == ["13812345678", "15987654321"]
        return ["字符串长度约11位", "长度一致", "数字字符占比100%"]


def _field() -> FieldProfile:
    return FieldProfile(
        field_name="contact_value",
        field_cn="联系值",
        sample_values=["13812345678", "15987654321"],
    )


def _card() -> SemanticCard:
    return SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号", "phone"],
        common_field_names=["phone", "mobile"],
        value_features=["通常由数字组成"],
        description="用于联系自然人的电话号码",
        semantic_category=["联系方式", "个人信息"],
        regulation_keywords=["手机号码", "电话号码", "联系方式"],
    )


def test_objective_profile_never_calls_semantic_candidate_rules():
    profile = build_objective_profile(_field(), ObjectiveOnlyProfiler())

    assert profile.features == ["字符串长度约11位", "长度一致", "数字字符占比100%"]
    assert "candidate_types" not in profile.model_dump()


def test_semantic_query_contains_names_samples_and_objective_features():
    profile = build_objective_profile(_field(), ObjectiveOnlyProfiler())

    query = build_semantic_query(_field(), profile)

    assert "字段名称：contact_value" in query
    assert "中文名称：联系值" in query
    assert "样本值：13812345678、15987654321" in query
    assert "字段客观特征：字符串长度约11位；长度一致；数字字符占比100%" in query


def test_regulation_query_uses_semantics_and_excludes_raw_samples_and_profile():
    query = build_regulation_bridge_query(_field(), _card())

    assert "字段：contact_value" in query
    assert "字段语义类型：手机号码" in query
    assert "语义类别：联系方式、个人信息" in query
    assert "法规检索关键词：手机号码、电话号码、联系方式" in query
    assert "13812345678" not in query
    assert "字符串长度约11位" not in query

