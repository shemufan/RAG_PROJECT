# Profile Query Submodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add C1 and C2 profile-query ablations to the CSV experiment CLI while preserving the current C query and all non-query experiment behavior.

**Architecture:** Keep one `RetrievalQueryBuilder` and let a small `ProfileQueryMode` control whether its already-computed `features` and `candidate_types` sections are emitted. Pass the mode through the existing builder factory and classification service only from `scripts/run_csv_pipeline.py`; every other caller receives the current full-profile default.

**Tech Stack:** Python 3.10+, standard-library `argparse` and typing, Pydantic field/value schemas, pytest, Ruff.

---

## File map

- Modify `app/rag/retrieval_query.py`: define `c/c1/c2` and selectively emit the two profile sections without duplicating profiling logic.
- Modify `app/services/classification_service.py`: pass an optional profile mode into the existing builder factory.
- Modify `scripts/run_csv_pipeline.py`: expose and validate `--query-mode` only for CSV experiments.
- Modify `tests/unit/test_retrieval_query.py`: verify exact C/C1/C2 Query composition and profiler reuse.
- Modify `tests/unit/test_services.py`: verify unchanged retrieval, `k=3`, Prompt and LLM flow for all profile modes.
- Modify `tests/integration/test_csv_pipeline_cli.py`: verify CLI defaults, choices, invalid combinations and propagation.

## Task 1: Extend the existing profile Query builder

**Files:**

- Modify: `tests/unit/test_retrieval_query.py`
- Modify: `app/rag/retrieval_query.py`

- [ ] **Step 1: Add failing exact-composition tests**

Add a parametrized test using the existing `StaticProfiler` and `noisy_field()` helpers:

```python
@pytest.mark.parametrize(
    ("mode", "has_features", "has_candidates"),
    [
        ("c", True, True),
        ("c1", True, False),
        ("c2", False, True),
    ],
)
def test_profile_submode_controls_only_enriched_sections(
    mode: str,
    has_features: bool,
    has_candidates: bool,
):
    profiler = StaticProfiler(
        ValueProfile(
            features=["6组十六进制字符", "冒号分隔"],
            candidate_types=["MAC地址", "设备标识信息"],
        )
    )
    builder = RetrievalQueryBuilder(
        value_profiler=profiler,
        profile_mode=mode,
    )

    query = builder.build(noisy_field())

    assert "字段名：device_attr" in query
    assert "样例值：A1:B2:C3:D4:E5:F6、11:22:33:44:55:66" in query
    assert ("数据结构特征：" in query) is has_features
    assert ("候选数据类型：" in query) is has_candidates
```

Extend `StaticProfiler` with `call_count`, increment it in `profile()`, and assert `call_count == 1` for every mode. Add a direct-construction test asserting an unknown mode raises `ValueError` with `unsupported profile query mode`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
```

Expected: new tests fail because `RetrievalQueryBuilder` does not accept `profile_mode`.

- [ ] **Step 3: Add the mode type and section flags**

In `app/rag/retrieval_query.py`, extend the typing declarations:

```python
QueryStrategy = Literal["legacy", "clean", "profile"]
ProfileQueryMode = Literal["c", "c1", "c2"]

_PROFILE_SECTIONS: dict[str, tuple[bool, bool]] = {
    "c": (True, True),
    "c1": (True, False),
    "c2": (False, True),
}
```

Extend the existing builder constructor without changing its profiler creation:

```python
def __init__(
    self,
    value_profiler: ValueProfilerProtocol | None = None,
    *,
    profile_mode: ProfileQueryMode = "c",
) -> None:
    if value_profiler is None:
        from app.services.value_profiler import ValueProfiler

        value_profiler = ValueProfiler()
    try:
        include_features, include_candidates = _PROFILE_SECTIONS[profile_mode]
    except KeyError as exc:
        raise ValueError(
            f"unsupported profile query mode {profile_mode!r}; choose from c, c1, c2"
        ) from exc
    self.value_profiler = value_profiler
    self.include_features = include_features
    self.include_candidates = include_candidates
```

Change only the two optional section conditions in `build()`:

```python
if self.include_features and value_profile.features:
    parts.append(f"数据结构特征：{'、'.join(value_profile.features)}")
if self.include_candidates and value_profile.candidate_types:
    parts.append(
        f"候选数据类型：{'、'.join(value_profile.candidate_types)}"
    )
```

The call to `ValueProfiler.profile()`, field name, representative samples, formatting and ordering remain unchanged.

- [ ] **Step 4: Pass the mode through the factory**

Change the factory signature and instantiate only the existing profile builder with the mode:

```python
def create_query_builder(
    strategy: str,
    *,
    profile_mode: ProfileQueryMode = "c",
) -> QueryBuilder:
    try:
        builder_factory = _QUERY_BUILDERS[strategy]
    except KeyError as exc:
        choices = ", ".join(_QUERY_BUILDERS)
        raise ValueError(
            f"unsupported query strategy {strategy!r}; choose from {choices}"
        ) from exc
    if strategy == "profile":
        return RetrievalQueryBuilder(profile_mode=profile_mode)
    return builder_factory()
```

Add a factory test proving `create_query_builder("profile", profile_mode="c1")` returns a profile builder whose Query excludes candidate types. Existing factory tests for `legacy`, `clean` and `profile` stay unchanged.

- [ ] **Step 5: Run focused verification**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py -q
python -m ruff check app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
```

Expected: all builder tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit the builder change**

```powershell
git add app/rag/retrieval_query.py tests/unit/test_retrieval_query.py
git commit -m "feat: add profile query submodes"
```

## Task 2: Pass the submode through the classification service

**Files:**

- Modify: `tests/unit/test_services.py`
- Modify: `app/services/classification_service.py`

- [ ] **Step 1: Add failing service-flow tests**

Extend the existing strategy retrieval test with a separate parametrized test for `c`, `c1`, and `c2`. For each mode, construct:

```python
service = FieldClassificationService(
    store,
    llm,
    query_strategy="profile",
    profile_query_mode=mode,
)
```

Classify the same MAC-like field and assert:

```python
assert store.k == 3
assert ("数据结构特征：" in store.query) is has_features
assert ("候选数据类型：" in store.query) is has_candidates
assert "设备标识规则" in llm.prompt[1].content
assert result.decision_path == "rag_llm"
```

Keep the existing injected-Builder test and all legacy/clean/profile tests intact.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/unit/test_services.py -q
```

Expected: new tests fail because `FieldClassificationService` does not accept `profile_query_mode`.

- [ ] **Step 3: Add the backward-compatible service argument**

Import `ProfileQueryMode`, add the keyword-only default, and pass it to the factory:

```python
def __init__(
    self,
    vector_store,
    llm_service,
    *,
    query_strategy: QueryStrategy = "profile",
    profile_query_mode: ProfileQueryMode = "c",
    query_builder: QueryBuilder | None = None,
):
    self.vector_store = vector_store
    self.llm_service = llm_service
    self.query_builder = (
        query_builder
        if query_builder is not None
        else create_query_builder(
            query_strategy,
            profile_mode=profile_query_mode,
        )
    )
```

Do not edit `classify_field()` or `build_query_text()`.

- [ ] **Step 4: Run service and compatibility tests**

Run:

```powershell
python -m pytest tests/unit/test_retrieval_query.py tests/unit/test_services.py tests/unit/test_database_pipeline.py -q
python -m ruff check app/services/classification_service.py tests/unit/test_services.py
```

Expected: all tests pass; callers that omit the new argument continue to use full C mode.

- [ ] **Step 5: Commit the service plumbing**

```powershell
git add app/services/classification_service.py tests/unit/test_services.py
git commit -m "refactor: pass profile query mode through classifier"
```

## Task 3: Expose C1 and C2 only in the CSV experiment CLI

**Files:**

- Modify: `tests/integration/test_csv_pipeline_cli.py`
- Modify: `scripts/run_csv_pipeline.py`

- [ ] **Step 1: Add failing CLI tests**

Add parser tests covering:

```python
@pytest.mark.parametrize("mode", ["c", "c1", "c2"])
def test_cli_accepts_profile_query_mode(tmp_path, mode: str):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(
        [
            "--input",
            str(path),
            "--query-strategy",
            "profile",
            "--query-mode",
            mode,
        ]
    )

    assert args.query_mode == mode
```

Also assert the default is `c`, an unknown mode raises `SystemExit`, and `--query-strategy clean --query-mode c1` raises `SystemExit`.

Update the existing pipeline-construction capture so it records both `query_strategy` and `profile_query_mode`; assert `build_pipeline(settings, "profile", "c2")` passes both values unchanged.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py -q
```

Expected: parser and construction tests fail because `--query-mode` and its propagation do not exist.

- [ ] **Step 3: Add and validate the CLI option**

Import `ProfileQueryMode` and add to `build_parser()`:

```python
parser.add_argument(
    "--query-mode",
    choices=("c", "c1", "c2"),
    default="c",
    help="profile Query submode: c=full, c1=features only, c2=candidates only",
)
```

After parsing, reject only explicit submodes paired with non-profile strategies:

```python
if args.query_strategy != "profile" and args.query_mode != "c":
    parser.error("--query-mode c1/c2 requires --query-strategy profile")
```

Extend pipeline construction:

```python
def build_pipeline(
    settings,
    query_strategy: QueryStrategy = "profile",
    profile_query_mode: ProfileQueryMode = "c",
) -> CSVClassificationPipeline:
```

Pass `profile_query_mode` into `FieldClassificationService`, then call:

```python
pipeline = build_pipeline(
    load_settings(),
    args.query_strategy,
    args.query_mode,
)
```

Do not edit summary formatting, CSV adapters, Benchmark repositories or metric calculations.

- [ ] **Step 4: Run CLI and adjacent pipeline tests**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py tests/integration/test_direct_csv_smoke.py -q
python -m pytest tests/unit/test_csv_pipeline.py tests/unit/test_database_pipeline.py tests/integration/test_benchmark_smoke.py tests/integration/test_pipeline_api.py -q
python -m ruff check scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
```

Expected: all tests pass and existing metric output remains unchanged.

- [ ] **Step 5: Commit the CLI option**

```powershell
git add scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
git commit -m "feat: expose profile query submodes in csv cli"
```

## Task 4: Verify experiment invariants and handoff commands

**Files:**

- Verify: `app/rag/retrieval_query.py`
- Verify: `app/services/classification_service.py`
- Verify: `scripts/run_csv_pipeline.py`
- Verify: all tests

- [ ] **Step 1: Run complete verification**

```powershell
python -m pytest -q
python -m ruff check .
python -m compileall -q app scripts tests
```

Expected: all tests pass, Ruff reports no errors and compileall exits zero.

- [ ] **Step 2: Smoke-print C, C1 and C2 Queries**

Run one local Python command against the same MAC-like `FieldProfile`:

```powershell
python -c "from app.rag.retrieval_query import RetrievalQueryBuilder; from app.schemas.field import FieldProfile; field=FieldProfile(field_name='device_attr', sample_values=['A1:B2:C3:D4:E5:F6']); [print(mode, RetrievalQueryBuilder(profile_mode=mode).build(field), sep='\n') for mode in ('c', 'c1', 'c2')]"
```

Expected:

- all three contain the identical field name and representative samples;
- `c` contains both enriched sections;
- `c1` contains only the feature section;
- `c2` contains only the candidate section.

- [ ] **Step 3: Audit the production diff**

```powershell
git diff --name-status 2520695..HEAD
git diff --check 2520695..HEAD
git status --short --branch
```

Expected production changes are limited to:

```text
app/rag/retrieval_query.py
app/services/classification_service.py
scripts/run_csv_pipeline.py
```

There must be no changes to Chroma, Embedding, Prompt, LLM, CSV adapters, Benchmark metrics, API, database Pipeline or `.env`.

- [ ] **Step 4: Record the two experiment commands**

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy profile --query-mode c1
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --label-column expected_personal --query-strategy profile --query-mode c2
```

Run both commands from the same branch and environment as the existing C experiment. Do not change `.env`, input, label column, model, Collection or other CLI arguments between runs.
