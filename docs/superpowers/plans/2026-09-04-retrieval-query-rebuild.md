# Retrieval Query Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Python value profiler and a clean retrieval-query builder, then route existing field classification retrieval through them without changing the LLM prompt or Chroma API.

**Architecture:** `ValueProfiler` converts only `field_name + sample_values` into a validated `ValueProfile`. `RetrievalQueryBuilder` selects at most three representative values and serializes the field name, structural features, and up to three candidate types. `FieldClassificationService` keeps its public `build_query_text()` method but delegates to these components before calling the existing `VectorStore.search(query, k=3)`.

**Tech Stack:** Python 3.10+, Pydantic v2, standard-library `re`/`ipaddress`/`uuid`/`urllib.parse`/`datetime`, pytest, Ruff.

---

## File map

- Create `app/services/value_profiler.py`: Pydantic result model, feature extraction, general format detectors, candidate ordering and cap.
- Create `app/rag/retrieval_query.py`: representative-value selection and deterministic single-line query serialization.
- Modify `app/services/classification_service.py`: dependency injection and delegation while preserving existing callers.
- Create `tests/unit/test_value_profiler.py`: required examples plus the complete first-version format matrix.
- Create `tests/unit/test_retrieval_query.py`: query content, exclusions, sample selection and empty-section behavior.
- Modify `tests/unit/test_services.py`: integration contract between profiler, query builder, vector retrieval and unchanged LLM prompt.

## Task 1: Introduce `ValueProfile` and the first required detectors

**Files:**

- Create: `tests/unit/test_value_profiler.py`
- Create: `app/services/value_profiler.py`

- [ ] **Step 1: Write failing tests for phone, MAC, masked email and ordinary product text**

Create `tests/unit/test_value_profiler.py` with the first behavior tests:

```python
import pytest

from app.services.value_profiler import ValueProfile, ValueProfiler


@pytest.fixture
def profiler() -> ValueProfiler:
    return ValueProfiler()


def test_profile_detects_masked_mobile_numbers(profiler: ValueProfiler):
    result = profiler.profile("attr_01", ["138**1234", "159**5678"])

    assert isinstance(result, ValueProfile)
    assert result.candidate_types[:2] == ["手机号码", "联系方式"]
    assert "存在脱敏字符" in result.features
    assert "多个样例格式一致" in result.features


def test_profile_detects_mac_from_value_with_meaningless_name(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["A1:B2:C3:D4:E5:F6"])

    assert result.candidate_types[:2] == ["MAC地址", "设备标识信息"]
    assert "6组十六进制字符" in result.features
    assert "冒号分隔" in result.features
    assert "格式稳定" in result.features


def test_profile_detects_masked_email(profiler: ValueProfiler):
    result = profiler.profile("attr_01", ["test***@example.com"])

    assert result.candidate_types[:2] == ["邮箱", "联系方式"]
    assert "包含@符号" in result.features
    assert "存在脱敏字符" in result.features


def test_profile_does_not_force_candidate_for_product_names(
    profiler: ValueProfiler,
):
    result = profiler.profile("product_name", ["无线鼠标", "机械键盘", "显示器"])

    assert result.candidate_types == []
    assert "主要由中文组成" in result.features


def test_profile_returns_empty_candidates_without_samples(profiler: ValueProfiler):
    result = profiler.profile("phone_number", ["", "   "])

    assert result.features == ["无有效样例"]
    assert result.candidate_types == []
```

- [ ] **Step 2: Run the new test module and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_value_profiler.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.services.value_profiler'`. This is the correct failure because no production module exists yet.

- [ ] **Step 3: Implement the Pydantic model, shared feature extraction and three exact detectors**

Create `app/services/value_profiler.py`:

```python
"""Explainable, rule-based profiling for field sample values."""

from __future__ import annotations

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
        return ValueProfile(
            features=features,
            candidate_types=candidates[:3],
        )

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
        joined = "".join(values)
        visible = [char for char in joined if not char.isspace()]
        if visible:
            counts = {
                "主要由数字组成": sum(char.isdigit() for char in visible),
                "主要由字母组成": sum(char.isalpha() and not _CHINESE_PATTERN.fullmatch(char) for char in visible),
                "主要由中文组成": sum(bool(_CHINESE_PATTERN.fullmatch(char)) for char in visible),
                "主要由十六进制字符组成": sum(char in "0123456789abcdefABCDEF" for char in visible),
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
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/unit/test_value_profiler.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Run Ruff on the new files**

Run:

```powershell
ruff check app/services/value_profiler.py tests/unit/test_value_profiler.py
```

Expected: `All checks passed!` Fix only formatting, imports, or line wrapping; rerun the five tests afterward.

- [ ] **Step 6: Commit the first profiler slice**

```powershell
git add app/services/value_profiler.py tests/unit/test_value_profiler.py
git commit -m "feat: profile core field value formats"
```

## Task 2: Complete the required first-version format matrix

**Files:**

- Modify: `tests/unit/test_value_profiler.py`
- Modify: `app/services/value_profiler.py`

- [ ] **Step 1: Add parameterized failing tests for every remaining required format**

Append to `tests/unit/test_value_profiler.py`:

```python
@pytest.mark.parametrize(
    ("field_name", "samples", "expected_candidate"),
    [
        ("column_x", ["192.168.1.10", "8.8.8.8"], "IPv4地址"),
        ("column_x", ["11010519491231002X"], "身份证号"),
        ("column_x", ["4532015112830366"], "银行卡号"),
        ("column_x", ["2026-09-04", "2026-09-05"], "日期时间"),
        (
            "column_x",
            ["550e8400-e29b-41d4-a716-446655440000"],
            "UUID",
        ),
        ("column_x", ["https://example.com/account?id=1"], "URL"),
        ("column_x", ["31.2304,121.4737"], "经纬度"),
        ("column_x", ["490154203237518"], "IMEI设备编号"),
    ],
)
def test_profile_detects_required_formats(
    profiler: ValueProfiler,
    field_name: str,
    samples: list[str],
    expected_candidate: str,
):
    result = profiler.profile(field_name, samples)

    assert expected_candidate in result.candidate_types


def test_profile_describes_plain_number_without_forcing_type(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["42", "108", "900"])

    assert "普通数字" in result.features
    assert result.candidate_types == []


def test_profile_describes_plain_text_without_forcing_type(
    profiler: ValueProfiler,
):
    result = profiler.profile("column_x", ["pending", "approved"])

    assert "普通文本" in result.features
    assert result.candidate_types == []


def test_profile_limits_and_deduplicates_candidates(profiler: ValueProfiler):
    result = profiler.profile("device_contact", ["490154203237518"])

    assert len(result.candidate_types) <= 3
    assert len(result.candidate_types) == len(set(result.candidate_types))
```

- [ ] **Step 2: Run only the added tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_value_profiler.py -q
```

Expected: the eight new format cases and ordinary-value assertions fail because their detectors and fallback features do not exist. The original five tests remain green.

- [ ] **Step 3: Add standard-library imports and detector patterns**

In `app/services/value_profiler.py`, add:

```python
import ipaddress
import uuid
from datetime import datetime
from urllib.parse import urlparse

_ID_CARD_PATTERN = re.compile(r"^(?:\d{17}[\dXx]|\d{6}[*xX]{8}\d{3}[\dXx])$")
_BANK_CARD_PATTERN = re.compile(r"^(?:\d{16,19}|\d{4}(?:[*xX]{4,11})\d{4})$")
_IMEI_PATTERN = re.compile(r"^\d{15}$")
_COORDINATE_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*$"
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)
```

- [ ] **Step 4: Add the remaining detector methods**

Add these methods to `ValueProfiler`:

```python
    @staticmethod
    def _is_ipv4(value: str) -> bool:
        try:
            return isinstance(ipaddress.ip_address(value), ipaddress.IPv4Address)
        except ValueError:
            return False

    @staticmethod
    def _is_id_card(value: str) -> bool:
        return bool(_ID_CARD_PATTERN.fullmatch(value))

    @staticmethod
    def _is_bank_card(value: str) -> bool:
        if not _BANK_CARD_PATTERN.fullmatch(value):
            return False
        if _MASK_PATTERN.search(value):
            return True
        return ValueProfiler._passes_luhn(value)

    @staticmethod
    def _is_datetime(value: str) -> bool:
        for date_format in _DATE_FORMATS:
            try:
                datetime.strptime(value, date_format)
            except ValueError:
                continue
            return True
        return False

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            return str(uuid.UUID(value)).lower() == value.lower()
        except ValueError:
            return False

    @staticmethod
    def _is_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _is_coordinate(value: str) -> bool:
        match = _COORDINATE_PATTERN.fullmatch(value)
        if not match:
            return False
        latitude, longitude = (float(item) for item in match.groups())
        return -90 <= latitude <= 90 and -180 <= longitude <= 180

    @staticmethod
    def _is_imei(value: str) -> bool:
        return bool(_IMEI_PATTERN.fullmatch(value)) and ValueProfiler._passes_luhn(value)

    @staticmethod
    def _passes_luhn(value: str) -> bool:
        total = 0
        parity = len(value) % 2
        for index, char in enumerate(value):
            digit = int(char)
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0
```

- [ ] **Step 5: Register detectors in unambiguous-first order**

Replace the local `detectors` tuple in `profile()` with:

```python
        detectors = (
            (self._is_mobile, ["手机号码", "联系方式"], ["符合手机号码结构"]),
            (self._is_email, ["邮箱", "联系方式"], ["符合邮箱结构"]),
            (self._is_ipv4, ["IPv4地址", "网络地址信息"], ["4段十进制数字", "点号分隔"]),
            (self._is_mac, ["MAC地址", "设备标识信息"], ["6组十六进制字符", "冒号分隔"]),
            (self._is_id_card, ["身份证号", "个人身份识别信息"], ["符合身份证号结构"]),
            (self._is_datetime, ["日期时间"], ["符合日期或时间结构"]),
            (self._is_uuid, ["UUID", "标识符"], ["符合UUID结构"]),
            (self._is_url, ["URL", "网络地址信息"], ["符合URL结构"]),
            (self._is_coordinate, ["经纬度", "位置信息"], ["两个十进制坐标", "逗号分隔"]),
            (self._is_imei, ["IMEI设备编号", "设备标识信息"], ["15位数字", "通过Luhn校验"]),
            (self._is_bank_card, ["银行卡号", "金融账户信息"], ["符合银行卡号结构"]),
        )
```

Use this order so exact delimiter-based formats are handled before length-overlapping numeric formats. Because the final list is capped, IMEI appears before the broader bank-card candidate for a valid 15-digit IMEI.

- [ ] **Step 6: Add fallback structure features only when no detector matches**

After the detector loop and before returning `ValueProfile`, add:

```python
        if not candidates:
            if all(value.isdigit() for value in values):
                self._extend_unique(features, ["普通数字"])
            elif all(any(char.isalpha() or _CHINESE_PATTERN.fullmatch(char) for char in value) for value in values):
                self._extend_unique(features, ["普通文本"])
```

No candidate is attached to either fallback.

- [ ] **Step 7: Run profiler tests and verify GREEN**

Run:

```powershell
python -m pytest tests/unit/test_value_profiler.py -q
ruff check app/services/value_profiler.py tests/unit/test_value_profiler.py
```

Expected: all profiler tests pass and Ruff reports no errors. If a test card number does not pass Luhn, replace the fixture with a documented Luhn-valid synthetic number rather than weakening the detector.

- [ ] **Step 8: Commit the complete format matrix**

```powershell
git add app/services/value_profiler.py tests/unit/test_value_profiler.py
git commit -m "feat: detect common field value structures"
```

## Task 3: Build concise retrieval queries with representative samples

**Files:**

- Create: `tests/unit/test_retrieval_query.py`
- Create: `app/rag/retrieval_query.py`

- [ ] **Step 1: Write failing tests for serialization, exclusions and sample selection**

Create `tests/unit/test_retrieval_query.py`:

```python
from app.rag.retrieval_query import RetrievalQueryBuilder
from app.services.value_profiler import ValueProfile


def test_builder_serializes_only_clean_retrieval_signals():
    query = RetrievalQueryBuilder().build(
        "device_attr",
        ["A1:B2:C3:D4:E5:F6"],
        ValueProfile(
            features=["6组十六进制字符", "冒号分隔", "格式稳定"],
            candidate_types=["MAC地址", "设备标识信息"],
        ),
    )

    assert query == (
        "字段名：device_attr "
        "样例值：A1:B2:C3:D4:E5:F6 "
        "数据结构特征：6组十六进制字符、冒号分隔、格式稳定 "
        "候选数据类型：MAC地址、设备标识信息"
    )


def test_builder_selects_at_most_three_distinct_representative_values():
    query = RetrievalQueryBuilder().build(
        "contact",
        ["138**1234", "138**1234", "159**5678", "test***@example.com", "010-12345678"],
        ValueProfile(features=["存在脱敏字符"]),
    )

    assert "138**1234、test***@example.com、010-12345678" in query
    assert "159**5678" not in query


def test_builder_omits_empty_optional_sections():
    query = RetrievalQueryBuilder().build(
        "product_name",
        [],
        ValueProfile(features=[], candidate_types=[]),
    )

    assert query == "字段名：product_name"
    assert "样例值" not in query
    assert "候选数据类型" not in query


def test_builder_cannot_receive_or_emit_field_profile_noise():
    query = RetrievalQueryBuilder().build(
        "attr_01",
        ["A1:B2:C3:D4:E5:F6"],
        ValueProfile(
            features=["6组十六进制字符"],
            candidate_types=["MAC地址"],
        ),
    )

    for forbidden in (
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
        "field_cn",
        "field_comment",
        "database_name",
        "table_name",
        "source_system",
    ):
        assert forbidden not in query
```

- [ ] **Step 2: Run query-builder tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.rag.retrieval_query'`.

- [ ] **Step 3: Implement stable format signatures and representative selection**

Create `app/rag/retrieval_query.py`:

```python
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
        distinct = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in distinct:
                distinct.append(cleaned)

        selected = []
        signatures = set()
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
        signature = []
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
```

The representative-value test deliberately treats both masked mobile values as the same structure. It keeps the first one, then chooses the differently shaped email and landline.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
ruff check app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
```

Expected: `4 passed` and no Ruff errors.

- [ ] **Step 5: Commit the builder**

```powershell
git add app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
git commit -m "feat: build clean retrieval queries"
```

## Task 4: Route existing classification retrieval through the new components

**Files:**

- Modify: `tests/unit/test_services.py`
- Modify: `app/services/classification_service.py`

- [ ] **Step 1: Update the existing service test to express the new contract**

In `tests/unit/test_services.py`, replace `test_classification_service_returns_structured_result` with:

```python
def test_classification_service_uses_profiled_retrieval_query_and_keeps_llm_prompt():
    store = FakeVectorStore(
        [
            Evidence(
                content="MAC地址属于设备标识信息。",
                source="个人信息安全规范.txt",
                article="附录A",
                score=0.91,
            )
        ]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="MAC地址",
            level="L3",
            confidence=0.92,
            reason="依据设备标识规则。",
            need_review=False,
        )
    )
    service = FieldClassificationService(store, llm)
    profile = FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="attr_01",
        field_cn="设备属性",
        field_comment="辅助说明",
        data_type="unknown",
        sample_values=["A1:B2:C3:D4:E5:F6"],
        business_domain="general",
    )

    result = service.classify_field(profile)

    assert "字段名：attr_01" in store.query
    assert "MAC地址" in store.query
    assert "设备标识信息" in store.query
    for forbidden in (
        "设备属性",
        "辅助说明",
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
    ):
        assert forbidden not in store.query
    assert '"field_cn": "设备属性"' in llm.prompt[1].content
    assert "MAC地址属于设备标识信息" in llm.prompt[1].content
    assert result.level == "L3"
    assert result.is_personal is True
    assert result.decision_path == "rag_llm"
```

Add an injection-focused test:

```python
def test_classification_service_build_query_text_delegates_to_injected_components():
    class StubProfiler:
        def profile(self, field_name: str, sample_values: list[str]):
            assert field_name == "column_x"
            assert sample_values == ["sample"]
            return ValueProfile(features=["stub feature"])

    class StubBuilder:
        def build(self, field_name: str, sample_values: list[str], value_profile):
            assert field_name == "column_x"
            assert sample_values == ["sample"]
            assert value_profile.features == ["stub feature"]
            return "clean query"

    service = FieldClassificationService(
        object(),
        object(),
        value_profiler=StubProfiler(),
        query_builder=StubBuilder(),
    )

    assert service.build_query_text(
        FieldProfile(field_name="column_x", sample_values=["sample"])
    ) == "clean query"
```

Also import `ValueProfile` at the top:

```python
from app.services.value_profiler import ValueProfile
```

- [ ] **Step 2: Run the two service tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_services.py `
  -k "profiled_retrieval_query or delegates_to_injected_components" -q
```

Expected failures:

- the existing `build_query_text()` still includes forbidden metadata and has no MAC candidates;
- `FieldClassificationService.__init__()` rejects the new keyword dependencies.

- [ ] **Step 3: Add imports and backward-compatible dependency injection**

In `app/services/classification_service.py`, add:

```python
from app.rag.retrieval_query import RetrievalQueryBuilder
from app.services.value_profiler import ValueProfiler
```

Replace the constructor with:

```python
    def __init__(
        self,
        vector_store,
        llm_service,
        *,
        value_profiler: ValueProfiler | None = None,
        query_builder: RetrievalQueryBuilder | None = None,
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.value_profiler = value_profiler or ValueProfiler()
        self.query_builder = query_builder or RetrievalQueryBuilder()
```

This preserves every existing two-positional-argument call site.

- [ ] **Step 4: Replace only the internals of `build_query_text()`**

Replace the current metadata dictionary with:

```python
    def build_query_text(self, field: FieldProfile) -> str:
        value_profile = self.value_profiler.profile(
            field.field_name,
            field.sample_values,
        )
        return self.query_builder.build(
            field.field_name,
            field.sample_values,
            value_profile,
        )
```

Do not modify `classify_field()` other than formatting caused by imports. The retrieval call remains `self.vector_store.search(self.build_query_text(profile), k=3)`, and the LLM still receives `build_classification_prompt(profile, evidence)`.

- [ ] **Step 5: Run service and new-module tests and verify GREEN**

Run:

```powershell
python -m pytest `
  tests/unit/test_value_profiler.py `
  tests/unit/test_retrieval_query.py `
  tests/unit/test_services.py -q
```

Expected: all selected tests pass, including the existing dependency-failure fallback test.

- [ ] **Step 6: Run adjacent Pipeline tests for interface compatibility**

Run:

```powershell
python -m pytest `
  tests/unit/test_csv_pipeline.py `
  tests/unit/test_database_pipeline.py `
  tests/integration/test_direct_csv_smoke.py `
  tests/integration/test_pipeline_api.py -q
```

Expected: all selected tests pass without changing Pipeline constructors or mocks.

- [ ] **Step 7: Commit the classification-service integration**

```powershell
git add app/services/classification_service.py tests/unit/test_services.py
git commit -m "refactor: use value profiles for retrieval queries"
```

## Task 5: Complete regression verification and requirement audit

**Files:**

- Verify: `app/services/value_profiler.py`
- Verify: `app/rag/retrieval_query.py`
- Verify: `app/services/classification_service.py`
- Verify: `tests/unit/test_value_profiler.py`
- Verify: `tests/unit/test_retrieval_query.py`
- Verify: `tests/unit/test_services.py`

- [ ] **Step 1: Compile all application and script modules**

Run:

```powershell
python -m compileall app scripts
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 2: Run the repository lint suite**

Run:

```powershell
ruff check .
```

Expected: `All checks passed!`.

- [ ] **Step 3: Run the complete automated test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass; the pre-implementation baseline was `158 passed, 2 skipped` and the final pass count must increase by the newly added tests.

- [ ] **Step 4: Run a direct query-content smoke check**

Run:

```powershell
python -c "from app.rag.retrieval_query import RetrievalQueryBuilder; from app.services.value_profiler import ValueProfiler; values=['A1:B2:C3:D4:E5:F6']; profile=ValueProfiler().profile('device_attr', values); print(RetrievalQueryBuilder().build('device_attr', values, profile))"
```

Expected output contains:

```text
字段名：device_attr
样例值：A1:B2:C3:D4:E5:F6
6组十六进制字符
MAC地址
设备标识信息
```

Expected output does not contain `csv_source`, `catalog_input`, `general`, `unknown`, `field_cn`, `field_comment`, `database_name`, `table_name` or `source_system`.

- [ ] **Step 5: Audit the implementation against every user requirement**

Use `Select-String` to confirm forbidden metadata is absent from the new builder and the old metadata map no longer exists in the classification service:

```powershell
Select-String -Path app/rag/retrieval_query.py,app/services/classification_service.py `
  -Pattern 'field_cn|field_comment|database_name|table_name|source_system|business_domain|data_type'
```

Expected: no matches. Then inspect `git diff feature/csv-input-adapters...HEAD -- app tests` and confirm only Query construction and its tests changed; LLM, Chroma, Pipeline and database modules must have no diff.

- [ ] **Step 6: Request code review and address findings**

Provide the reviewer with:

```text
Base: feature/csv-input-adapters@2e24639
Head: current query_rebuild HEAD
Scope: ValueProfiler, RetrievalQueryBuilder, classification-service Query delegation
Constraints: no LLM Prompt, Chroma, Pipeline or database behavior changes
```

Fix every Critical or Important issue with a failing regression test first. Rerun Tasks 5.1–5.5 after fixes.

- [ ] **Step 7: Record final status without creating an extra empty commit**

Run:

```powershell
git status --short --branch
git log --oneline --decorate feature/csv-input-adapters..HEAD
```

Expected: clean `query_rebuild` worktree and the design, plan, profiler, builder and integration commits listed in order.
