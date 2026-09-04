"""Build concise retrieval queries from field value profiles."""

from app.services.value_profiler import ValueProfile


class RetrievalQueryBuilder:
    """Serialize only signals intended for embedding retrieval."""

    def build(
        self,
        field_name: str,
        sample_values: list[str],
        value_profile: ValueProfile,
    ) -> str:
        parts = [f"字段名：{field_name.strip()}"]
        samples = self._representative_values(sample_values)
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
