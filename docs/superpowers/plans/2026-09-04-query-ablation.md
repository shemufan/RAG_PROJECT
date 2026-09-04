# Query Ablation Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `legacy|clean|profile` Query strategy switch to the CSV Benchmark CLI while keeping retrieval, LLM, Prompt, Pipeline, knowledge base and metrics behavior unchanged.

**Architecture:** Three builders implement one `build(field: FieldProfile) -> str` protocol. A small factory selects the builder once when `FieldClassificationService` is constructed, and `scripts/run_csv_pipeline.py` passes the CLI value into that service; all other callers rely on the default `profile` strategy.

**Tech Stack:** Python 3.10+, standard-library `argparse` and typing, Pydantic field schemas, pytest, Ruff.

---

## File map

- Modify `app/rag/retrieval_query.py`: unified protocol, legacy and clean builders, profile builder adaptation, strategy factory.
- Modify `app/services/classification_service.py`: construct or accept one builder and delegate Query creation to it.
- Modify `scripts/run_csv_pipeline.py`: parse and pass `--query-strategy`; no other CLI receives the flag.
- Modify `tests/unit/test_retrieval_query.py`: exact Query contracts, factory behavior and profile delegation.
- Modify `tests/unit/test_services.py`: verify all three strategies use the unchanged vector/LLM flow.
- Modify `tests/integration/test_csv_pipeline_cli.py`: CLI choices, default, rejection and service injection.

## Task 1: Unify the three Query builders

**Files:**

- Modify: `tests/unit/test_retrieval_query.py`
- Modify: `app/rag/retrieval_query.py`

- [ ] **Step 1: Replace old builder-call tests with the unified field interface**

Delete the old `test_builder_serializes_only_clean_retrieval_signals`; its exact
profile serialization contract is covered by the injected-profiler test below.
Update imports and add these exact contract tests in
`tests/unit/test_retrieval_query.py`:

```python
import pytest

from app.rag.retrieval_query import (
    CleanQueryBuilder,
    LegacyQueryBuilder,
    RetrievalQueryBuilder,
    create_query_builder,
)
from app.schemas.field import FieldProfile
from app.services.value_profiler import ValueProfile


def noisy_field() -> FieldProfile:
    return FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="device_attr",
        field_cn="设备属性",
        field_comment="采集终端设备属性",
        data_type="unknown",
        sample_values=["A1:B2:C3:D4:E5:F6", "11:22:33:44:55:66"],
        business_domain="general",
    )


def test_legacy_builder_matches_original_query_exactly():
    assert LegacyQueryBuilder().build(noisy_field()) == (
        "field_name: device_attr\n"
        "field_cn: 设备属性\n"
        "field_comment: 采集终端设备属性\n"
        "data_type: unknown\n"
        "sample_values: A1:B2:C3:D4:E5:F6、11:22:33:44:55:66\n"
        "business_domain: general\n"
        "table_name: catalog_input\n"
        "database_name: csv_source\n"
        "source_system: csv"
    )


def test_clean_builder_contains_only_original_core_values():
    query = CleanQueryBuilder().build(noisy_field())

    assert query == (
        "field_name: device_attr\n"
        "sample_values: A1:B2:C3:D4:E5:F6、11:22:33:44:55:66"
    )
    for forbidden in (
        "设备属性",
        "采集终端设备属性",
        "csv_source",
        "catalog_input",
        "general",
        "unknown",
    ):
        assert forbidden not in query


def test_profile_builder_calls_value_profiler():
    class StubProfiler:
        def profile(self, field_name: str, sample_values: list[str]):
            assert field_name == "device_attr"
            assert sample_values == [
                "A1:B2:C3:D4:E5:F6",
                "11:22:33:44:55:66",
            ]
            return ValueProfile(
                features=["6组十六进制字符", "冒号分隔"],
                candidate_types=["MAC地址", "设备标识信息"],
            )

    query = RetrievalQueryBuilder(value_profiler=StubProfiler()).build(noisy_field())

    assert "字段名：device_attr" in query
    assert "数据结构特征：6组十六进制字符、冒号分隔" in query
    assert "候选数据类型：MAC地址、设备标识信息" in query


@pytest.mark.parametrize(
    ("strategy", "builder_type"),
    [
        ("legacy", LegacyQueryBuilder),
        ("clean", CleanQueryBuilder),
        ("profile", RetrievalQueryBuilder),
    ],
)
def test_factory_returns_requested_builder(strategy: str, builder_type: type):
    assert isinstance(create_query_builder(strategy), builder_type)


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unsupported query strategy"):
        create_query_builder("other")
```

Use this small test helper for profile-query assertions that do not exercise real profiling:

```python
class StaticProfiler:
    def __init__(self, value_profile: ValueProfile):
        self.value_profile = value_profile

    def profile(self, field_name: str, sample_values: list[str]) -> ValueProfile:
        return self.value_profile
```

Replace the existing representative-value, empty-section and forbidden-noise tests with:

```python
def test_profile_builder_selects_at_most_three_representative_values():
    builder = RetrievalQueryBuilder(
        value_profiler=StaticProfiler(ValueProfile(features=["存在脱敏字符"]))
    )
    query = builder.build(
        FieldProfile(
            field_name="contact",
            sample_values=[
                "138**1234",
                "138**1234",
                "159**5678",
                "test***@example.com",
                "010-12345678",
            ],
        )
    )

    assert "138**1234、test***@example.com、010-12345678" in query
    assert "159**5678" not in query


def test_profile_builder_omits_empty_optional_sections():
    builder = RetrievalQueryBuilder(
        value_profiler=StaticProfiler(ValueProfile())
    )

    query = builder.build(FieldProfile(field_name="product_name"))

    assert query == "字段名：product_name"
    assert "样例值" not in query
    assert "候选数据类型" not in query


def test_profile_builder_does_not_emit_field_profile_noise():
    builder = RetrievalQueryBuilder(
        value_profiler=StaticProfiler(
            ValueProfile(
                features=["6组十六进制字符"],
                candidate_types=["MAC地址"],
            )
        )
    )

    query = builder.build(noisy_field())

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

Keep the fresh-process regression test exactly as follows:

```python
def test_retrieval_query_module_can_be_imported_first_in_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.rag.retrieval_query import RetrievalQueryBuilder",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Run the builder tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
```

Expected: collection or calls fail because `LegacyQueryBuilder`, `CleanQueryBuilder`, `create_query_builder`, and the unified `RetrievalQueryBuilder.build(field)` interface do not exist.

- [ ] **Step 3: Implement the unified protocol and raw builders**

At the top of `app/rag/retrieval_query.py`, use these imports and types:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

from app.schemas.field import FieldProfile

if TYPE_CHECKING:
    from app.services.value_profiler import ValueProfile, ValueProfiler

QueryStrategy = Literal["legacy", "clean", "profile"]


class QueryBuilder(Protocol):
    def build(self, field: FieldProfile) -> str: ...


class ValueProfilerProtocol(Protocol):
    def profile(
        self,
        field_name: str,
        sample_values: list[str],
    ) -> ValueProfile: ...


class LegacyQueryBuilder:
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
    def build(self, field: FieldProfile) -> str:
        values = {
            "field_name": field.field_name,
            "sample_values": "、".join(field.sample_values),
        }
        return "\n".join(
            f"{key}: {value}" for key, value in values.items() if value
        )
```

- [ ] **Step 4: Adapt `RetrievalQueryBuilder` to own profiling and accept a field**

Replace its public constructor/build method with:

```python
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
```

Keep `_representative_values()` and `_format_signature()` unchanged. The local `ValueProfiler` import is required: importing it at module load time would recreate the package initialization cycle previously covered by the fresh-process regression test.

- [ ] **Step 5: Add the strategy factory**

Append:

```python
_QUERY_BUILDERS: dict[str, Callable[[], QueryBuilder]] = {
    "legacy": LegacyQueryBuilder,
    "clean": CleanQueryBuilder,
    "profile": RetrievalQueryBuilder,
}


def create_query_builder(strategy: str) -> QueryBuilder:
    try:
        builder_factory = _QUERY_BUILDERS[strategy]
    except KeyError as exc:
        choices = ", ".join(_QUERY_BUILDERS)
        raise ValueError(
            f"unsupported query strategy {strategy!r}; choose from {choices}"
        ) from exc
    return builder_factory()
```

- [ ] **Step 6: Run focused tests and lint**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
python -m ruff check app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
```

Expected: all Query tests pass, including the fresh-process import test, and Ruff reports `All checks passed!`.

- [ ] **Step 7: Commit the Builder layer**

```powershell
git add app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
git commit -m "feat: add retrieval query strategies"
```

## Task 2: Select one Builder in the classification service

**Files:**

- Modify: `tests/unit/test_services.py`
- Modify: `app/services/classification_service.py`

- [ ] **Step 1: Add service tests for all three strategies and unchanged top-k**

Make `FakeVectorStore.search()` record `self.k = k`. Add:

```python
import pytest


@pytest.mark.parametrize(
    ("strategy", "expected_query_fragment"),
    [
        ("legacy", "field_cn: 设备属性"),
        ("clean", "field_name: attr_01"),
        ("profile", "候选数据类型：MAC地址、设备标识信息"),
    ],
)
def test_classification_service_retrieves_with_each_query_strategy(
    strategy: str,
    expected_query_fragment: str,
):
    store = FakeVectorStore(
        [Evidence(content="设备标识规则", source="rules.md", score=0.9)]
    )
    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人常用设备信息",
            subcategory="MAC地址",
            level="L3",
            confidence=0.9,
            reason="设备标识规则",
            need_review=False,
        )
    )
    service = FieldClassificationService(store, llm, query_strategy=strategy)

    result = service.classify_field(
        FieldProfile(
            field_name="attr_01",
            field_cn="设备属性",
            sample_values=["A1:B2:C3:D4:E5:F6"],
        )
    )

    assert expected_query_fragment in store.query
    assert store.k == 3
    assert "设备标识规则" in llm.prompt[1].content
    assert result.decision_path == "rag_llm"
```

Replace the existing injected-components test with the unified interface:

```python
def test_classification_service_delegates_to_injected_query_builder():
    class StubBuilder:
        def build(self, field: FieldProfile) -> str:
            assert field.field_name == "column_x"
            assert field.sample_values == ["sample"]
            return "clean query"

    service = FieldClassificationService(
        object(),
        object(),
        query_builder=StubBuilder(),
    )

    query = service.build_query_text(
        FieldProfile(field_name="column_x", sample_values=["sample"])
    )

    assert query == "clean query"
```

- [ ] **Step 2: Run service tests and verify RED**

Run:

```powershell
python -m pytest tests/unit/test_services.py -q
```

Expected: failures because the service does not accept `query_strategy`, still owns a profiler, and invokes the old three-argument builder interface.

- [ ] **Step 3: Replace profiler ownership with factory-selected Builder ownership**

In `app/services/classification_service.py`, import:

```python
from app.rag.retrieval_query import (
    QueryBuilder,
    QueryStrategy,
    create_query_builder,
)
```

Remove the direct `ValueProfiler` and `RetrievalQueryBuilder` imports. Replace the constructor and `build_query_text()` with:

```python
    def __init__(
        self,
        vector_store,
        llm_service,
        *,
        query_strategy: QueryStrategy = "profile",
        query_builder: QueryBuilder | None = None,
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.query_builder = (
            query_builder
            if query_builder is not None
            else create_query_builder(query_strategy)
        )

    def build_query_text(self, field: FieldProfile) -> str:
        return self.query_builder.build(field)
```

Do not edit the body of `classify_field()`.

- [ ] **Step 4: Run service and adjacent tests**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py tests/unit/test_services.py -q
python -m ruff check app/rag/retrieval_query.py app/services/classification_service.py tests/unit/test_retrieval_query.py tests/unit/test_services.py
```

Expected: all selected tests and Ruff pass. The existing default-profile test proves API/database callers that use the two positional arguments retain current behavior.

- [ ] **Step 5: Commit service selection**

```powershell
git add app/services/classification_service.py tests/unit/test_services.py
git commit -m "refactor: select retrieval query builder by strategy"
```

## Task 3: Expose the strategy only in the CSV Pipeline CLI

**Files:**

- Modify: `tests/integration/test_csv_pipeline_cli.py`
- Modify: `scripts/run_csv_pipeline.py`

- [ ] **Step 1: Add parser tests for default, valid and invalid values**

Add to `tests/integration/test_csv_pipeline_cli.py`:

```python
@pytest.mark.parametrize("strategy", ["legacy", "clean", "profile"])
def test_cli_accepts_query_strategy(tmp_path, strategy: str):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(["--input", str(path), "--query-strategy", strategy])

    assert args.query_strategy == strategy


def test_cli_defaults_query_strategy_to_profile(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).query_strategy == "profile"


def test_cli_rejects_unknown_query_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--query-strategy", "other"])
```

- [ ] **Step 2: Add a pipeline-construction test for service injection**

Add `from types import SimpleNamespace` and `import scripts.run_csv_pipeline as csv_cli`, then add:

```python
def test_build_pipeline_passes_query_strategy_to_classifier(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def __init__(self, embedding_service, *, settings):
            pass

        def count(self) -> int:
            return 1

    class CapturingClassifier:
        def __init__(self, vector_store, llm_service, *, query_strategy):
            captured["query_strategy"] = query_strategy

    monkeypatch.setattr(csv_cli, "EmbeddingService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(csv_cli, "LLMService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "FieldClassificationService", CapturingClassifier)
    monkeypatch.setattr(csv_cli, "BenchmarkTargetRepository", lambda url: object())
    settings = SimpleNamespace(
        target_database_url="sqlite://",
        embedding_model_path="model",
        deepseek_model="llm",
        knowledge_base_version="kb",
    )

    csv_cli.build_pipeline(settings, query_strategy="clean")

    assert captured["query_strategy"] == "clean"
```

- [ ] **Step 3: Run CLI tests and verify RED**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py -q
```

Expected: parser tests fail because the flag does not exist, and the construction test fails because `build_pipeline()` does not accept the strategy.

- [ ] **Step 4: Add the parser option and pass it to the service**

In `scripts/run_csv_pipeline.py`, import `QueryStrategy` and add to `build_parser()`:

```python
    parser.add_argument(
        "--query-strategy",
        choices=("legacy", "clean", "profile"),
        default="profile",
        help=(
            "retrieval Query strategy; use the same value when resuming a run"
        ),
    )
```

Change `build_pipeline()` to:

```python
def build_pipeline(
    settings,
    query_strategy: QueryStrategy = "profile",
) -> CSVClassificationPipeline:
```

Pass the value into the existing service construction:

```python
    classifier = FieldClassificationService(
        vector_store,
        LLMService(settings=settings),
        query_strategy=query_strategy,
    )
```

Finally change the one call in `main()` to:

```python
    pipeline = build_pipeline(load_settings(), args.query_strategy)
```

Do not modify any other script.

- [ ] **Step 5: Run CLI, direct CSV and compatibility tests**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py tests/integration/test_direct_csv_smoke.py -q
python -m pytest tests/unit/test_csv_pipeline.py tests/unit/test_database_pipeline.py tests/integration/test_benchmark_smoke.py tests/integration/test_pipeline_api.py -q
python -m ruff check scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
```

Expected: all selected tests and Ruff pass. Existing calls without a strategy continue to receive `profile`.

- [ ] **Step 6: Commit the CLI switch**

```powershell
git add scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
git commit -m "feat: expose query strategy in csv pipeline cli"
```

## Task 4: Final experiment-invariant audit

**Files:**

- Verify: `app/rag/retrieval_query.py`
- Verify: `app/services/classification_service.py`
- Verify: `scripts/run_csv_pipeline.py`
- Verify: all tests

- [ ] **Step 1: Compile and lint the repository**

Run:

```powershell
python -m compileall -q app scripts
python -m ruff check .
```

Expected: both commands exit 0 and Ruff prints `All checks passed!`.

- [ ] **Step 2: Run the complete test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass, the two existing opt-in tests remain skipped, and the pass count is greater than the `185 passed, 2 skipped` branch baseline.

- [ ] **Step 3: Smoke-print all three Queries from one field**

Run:

```powershell
python -c "from app.rag.retrieval_query import create_query_builder; from app.schemas.field import FieldProfile; field=FieldProfile(field_name='device_attr',field_cn='设备属性',sample_values=['A1:B2:C3:D4:E5:F6']); [print(name, create_query_builder(name).build(field), sep='\n') for name in ('legacy','clean','profile')]"
```

Expected:

- legacy contains `field_cn` and the FieldProfile defaults;
- clean contains only `field_name` and `sample_values`;
- profile contains MAC structural features and candidates.

- [ ] **Step 4: Audit the only changed production files**

Run:

```powershell
git diff --name-status query_rebuild...HEAD
git diff --check query_rebuild...HEAD
git status --short --branch
```

Expected production changes are limited to:

```text
app/rag/retrieval_query.py
app/services/classification_service.py
scripts/run_csv_pipeline.py
```

plus the design, plan and related tests. There must be no diff in `app/repositories/vector_store.py`, `app/rag/prompt.py`, `app/services/llm_service.py`, `app/services/csv_pipeline.py`, database Pipeline modules or Benchmark metric calculation.

- [ ] **Step 5: Record the three actual experiment commands for handoff**

Use the same input path and label column in all commands; only the strategy changes:

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy legacy
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy clean
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy profile
```

Do not run paid or external experiments unless the user separately authorizes them. The implementation verification uses offline tests only.

- [ ] **Step 6: Inspect final branch status without an empty commit**

Run:

```powershell
git log --oneline --decorate query_rebuild..HEAD
git status --short --branch
```

Expected: the design, plan, Builder, service and CLI commits appear in order, and the worktree is clean on `experiment/query-ablation`.
