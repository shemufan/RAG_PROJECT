"""Deterministic queries for the two-stage Semantic Bridge experiment."""

from typing import Protocol

from app.schemas.field import FieldProfile
from app.schemas.semantic import ObjectiveValueProfile, SemanticCard


class ObjectiveProfiler(Protocol):
    def basic_statistics(self, sample_values: list[str]) -> list[str]: ...


def build_objective_profile(
    field: FieldProfile,
    profiler: ObjectiveProfiler,
) -> ObjectiveValueProfile:
    """Extract observable value facts without invoking semantic detectors."""

    return ObjectiveValueProfile(
        features=profiler.basic_statistics(field.sample_values)
    )


def build_semantic_query(
    field: FieldProfile,
    profile: ObjectiveValueProfile,
) -> str:
    """Describe field shape for retrieval against Semantic Cards."""

    parts = [f"字段名称：{field.field_name}"]
    if field.field_cn:
        parts.append(f"中文名称：{field.field_cn}")
    samples = [value.strip() for value in field.sample_values if value.strip()]
    if samples:
        parts.append(f"样本值：{'、'.join(samples)}")
    if profile.features:
        parts.append(f"字段客观特征：{'；'.join(profile.features)}")
    return "\n".join(parts)


def build_regulation_bridge_query(
    field: FieldProfile,
    card: SemanticCard,
) -> str:
    """Bridge selected domain semantics to regulatory language."""

    parts = [f"字段：{field.field_name}"]
    if field.field_cn:
        parts.append(f"中文名称：{field.field_cn}")
    parts.extend(
        (
            f"字段语义类型：{card.semantic_type}",
            f"语义类别：{'、'.join(card.semantic_category)}",
            f"法规检索关键词：{'、'.join(card.regulation_keywords)}",
        )
    )
    return "\n".join(parts)
