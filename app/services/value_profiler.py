"""Explainable, rule-based profiling for field sample values."""

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

_MASK_PATTERN: re.Pattern[str] = re.compile(r"(?:\*+|[xX]{2,})")
_CHINESE_PATTERN: re.Pattern[str] = re.compile(r"[\u4e00-\u9fff]")
_MOBILE_PATTERN: re.Pattern[str] = re.compile(
    r"^1[3-9]\d(?:\d{8}|(?:\*{2,4}|[xX]{2,4})\d{4})$"
)
_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"^[^@\s]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}$"
)
_MAC_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$"
)


class ValueProfile(BaseModel):
    """Validated structural summary used only for retrieval."""

    features: list[str] = Field(default_factory=list)
    candidate_types: list[str] = Field(default_factory=list, max_length=3)


class ValueProfiler:
    """Infer value shape and cautious candidates without using an LLM."""

    def profile(self, field_name: str, sample_values: list[str]) -> ValueProfile:
        values = [value.strip() for value in sample_values if value.strip()]
        if not values:
            return ValueProfile(features=["无有效样例"])

        features = self._general_features(values)
        candidates: list[str] = []
        detectors: tuple[
            tuple[Callable[[str], bool], list[str], list[str]], ...
        ] = (
            (
                self._is_mobile,
                ["手机号码", "联系方式"],
                ["符合手机号码结构"],
            ),
            (
                self._is_email,
                ["邮箱", "联系方式"],
                ["符合邮箱结构"],
            ),
            (
                self._is_mac,
                ["MAC地址", "设备标识信息"],
                ["6组十六进制字符", "冒号分隔"],
            ),
        )
        for detector, detected_candidates, detected_features in detectors:
            if all(detector(value) for value in values):
                self._extend_unique(features, detected_features)
                self._extend_unique(candidates, detected_candidates)

        candidates = self._prioritize_with_field_name(field_name, candidates)
        return ValueProfile(features=features, candidate_types=candidates[:3])

    @staticmethod
    def _general_features(values: list[str]) -> list[str]:
        lengths = [len(value) for value in values]
        typical_length = round(sum(lengths) / len(lengths))
        features = [f"字符串长度约{typical_length}位"]
        if len(set(lengths)) == 1:
            features.append("长度一致")
        if len(values) > 1 and len({_format_signature(value) for value in values}) == 1:
            features.append("多个样例格式一致")
        else:
            features.append("格式稳定")

        visible = [char for value in values for char in value if not char.isspace()]
        if visible:
            counts = {
                "主要由数字组成": sum(char.isdigit() for char in visible),
                "主要由字母组成": sum(
                    char.isalpha() and not _CHINESE_PATTERN.fullmatch(char)
                    for char in visible
                ),
                "主要由中文组成": sum(
                    bool(_CHINESE_PATTERN.fullmatch(char)) for char in visible
                ),
                "主要由十六进制字符组成": sum(
                    char in "0123456789abcdefABCDEF" for char in visible
                ),
            }
            label, count = max(counts.items(), key=lambda item: item[1])
            if count / len(visible) >= 0.6:
                features.append(label)
        if any(_MASK_PATTERN.search(value) for value in values):
            features.append("存在脱敏字符")
        if any("@" in value for value in values):
            features.append("包含@符号")
        return features

    @staticmethod
    def _extend_unique(target: list[str], values: list[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)

    @staticmethod
    def _prioritize_with_field_name(
        field_name: str,
        candidates: list[str],
    ) -> list[str]:
        """Use generic name hints only to order candidates found from values."""

        normalized = field_name.casefold()
        hints = (
            (("phone", "mobile", "email", "contact"), "联系方式"),
            (("device", "mac", "imei"), "设备标识信息"),
            (("bank", "card", "account"), "金融账户信息"),
            (("latitude", "longitude", "location", "geo"), "位置信息"),
            (("identity", "id_card", "passport"), "个人身份识别信息"),
        )
        preferred = next(
            (
                candidate
                for tokens, candidate in hints
                if any(token in normalized for token in tokens)
                and candidate in candidates
            ),
            None,
        )
        if preferred is None or candidates.index(preferred) <= 1:
            return candidates
        return [
            candidates[0],
            preferred,
            *(item for item in candidates[1:] if item != preferred),
        ]

    @staticmethod
    def _is_mobile(value: str) -> bool:
        return bool(_MOBILE_PATTERN.fullmatch(value))

    @staticmethod
    def _is_email(value: str) -> bool:
        return bool(_EMAIL_PATTERN.fullmatch(value))

    @staticmethod
    def _is_mac(value: str) -> bool:
        return bool(_MAC_PATTERN.fullmatch(value))


def _format_signature(value: str) -> str:
    """Collapse characters into stable classes while preserving separators."""

    classes = []
    previous = ""
    for char in value:
        if char.isdigit():
            current = "D"
        elif _CHINESE_PATTERN.fullmatch(char):
            current = "C"
        elif char in "*xX":
            current = "M"
        elif char.isalpha():
            current = "A"
        else:
            current = char
        if current != previous or current not in {"D", "C", "A", "M"}:
            classes.append(current)
        previous = current
    return "".join(classes)
