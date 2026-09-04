"""Build selectable retrieval queries from validated field profiles."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

from app.schemas.field import FieldProfile

if TYPE_CHECKING:
    from app.services.value_profiler import ValueProfile

QueryStrategy = Literal["legacy", "clean", "profile"]


class QueryBuilder(Protocol):
    """Common interface for retrieval Query strategies."""

    def build(self, field: FieldProfile) -> str: ...


class ValueProfilerProtocol(Protocol):
    """Structural interface required by the profile Query strategy."""

    def profile(
        self,
        field_name: str,
        sample_values: list[str],
    ) -> ValueProfile: ...


class LegacyQueryBuilder:
    """Reproduce the original full-metadata retrieval Query."""

    def build(self, field: FieldProfile) -> str:
        values = {
            "field_name": field.field_name,
            "field_cn": field.field_cn,
            "field_comment": field.field_comment,
            "data_type": field.data_type,
            "sample_values": "、".join(field.sample_values),
            "business_domain": field.business_domain,
            "table_name": field.table_name,
            "database_name": field.database_name,
            "source_system": field.source_system,
        }
        return "\n".join(
            f"{key}: {value}" for key, value in values.items() if value
        )


class CleanQueryBuilder:
    """Build a Query from only the original field name and sample values."""

    def build(self, field: FieldProfile) -> str:
        values = {
            "field_name": field.field_name,
            "sample_values": "、".join(field.sample_values),
        }
        return "\n".join(
            f"{key}: {value}" for key, value in values.items() if value
        )


class RetrievalQueryBuilder:
    """Build the profile-enriched retrieval Query."""

    def __init__(
        self,
        value_profiler: ValueProfilerProtocol | None = None,
    ) -> None:
        if value_profiler is None:
            from app.services.value_profiler import ValueProfiler

            value_profiler = ValueProfiler()
        self.value_profiler = value_profiler

    def build(self, field: FieldProfile) -> str:
        value_profile = self.value_profiler.profile(
            field.field_name,
            field.sample_values,
        )
        parts = [f"字段名：{field.field_name}"]
        samples = self._representative_values(field.sample_values)
        if samples:
            parts.append(f"样例值：{'、'.join(samples)}")
        if value_profile.features:
            parts.append(f"数据结构特征：{'、'.join(value_profile.features)}")
        if value_profile.candidate_types:
            parts.append(
                f"候选数据类型：{'、'.join(value_profile.candidate_types)}"
            )
        return " ".join(parts)

    @classmethod
    def _representative_values(cls, values: list[str]) -> list[str]:
        distinct: list[str] = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in distinct:
                distinct.append(cleaned)

        selected: list[str] = []
        signatures: set[str] = set()
        for value in distinct:
            signature = cls._format_signature(value)
            if signature not in signatures:
                selected.append(value)
                signatures.add(signature)
            if len(selected) == 3:
                return selected

        for value in distinct:
            if value not in selected:
                selected.append(value)
            if len(selected) == 3:
                break
        return selected

    @staticmethod
    def _format_signature(value: str) -> str:
        signature: list[str] = []
        previous = ""
        for char in value:
            if char.isdigit():
                current = "D"
            elif "\u4e00" <= char <= "\u9fff":
                current = "C"
            elif char in "*xX":
                current = "M"
            elif char.isalpha():
                current = "A"
            else:
                current = char
            if current != previous or current not in {"D", "C", "A", "M"}:
                signature.append(current)
            previous = current
        return "".join(signature)


_QUERY_BUILDERS: dict[str, Callable[[], QueryBuilder]] = {
    "legacy": LegacyQueryBuilder,
    "clean": CleanQueryBuilder,
    "profile": RetrievalQueryBuilder,
}


def create_query_builder(strategy: str) -> QueryBuilder:
    """Create one Query builder or reject an unsupported strategy."""

    try:
        builder_factory = _QUERY_BUILDERS[strategy]
    except KeyError as exc:
        choices = ", ".join(_QUERY_BUILDERS)
        raise ValueError(
            f"unsupported query strategy {strategy!r}; choose from {choices}"
        ) from exc
    return builder_factory()
