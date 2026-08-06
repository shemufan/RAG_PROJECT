# Multi-mode CSV Input Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify either catalog-style or ordinary tabular CSV files directly through the existing `FieldProfile` → RAG → structured LLM path, with optional leakage-safe labels, B persistence, and resumable execution.

**Architecture:** A shared streaming CSV reader feeds deterministic mode detection and two focused adapters that emit one canonical `CSVInputBatch`. A new CSV pipeline passes only each case's `FieldProfile` to `FieldClassificationService`; labels stay outside model input. Existing MySQL and Benchmark import paths remain intact in this phase, while B benchmark tables are extended compatibly for direct CSV runs.

**Tech Stack:** Python 3.10, Pydantic, SQLAlchemy Core, MySQL 8, FastAPI, pytest, Ruff.

---

## File map

**Create**

- `app/schemas/csv_input.py`: input modes, normalized CSV cases, batches, limits, and label summaries.
- `app/services/csv_reader.py`: size/encoding/header validation and streaming row access.
- `app/services/csv_mode_detector.py`: deterministic `auto` mode selection.
- `app/services/catalog_csv_adapter.py`: row-oriented catalog conversion.
- `app/services/tabular_csv_adapter.py`: column-oriented business data conversion.
- `app/services/benchmark_label_service.py`: optional sidecar label parsing and matching.
- `app/services/csv_pipeline.py`: direct CSV classification, checkpointing, and resume validation.
- `scripts/run_csv_pipeline.py`: local new/resume command.
- `sql/migrations/2026-08-06_extend_benchmark_for_csv.sql`: compatible B migration.
- `tests/unit/test_csv_reader.py`
- `tests/unit/test_csv_mode_detector.py`
- `tests/unit/test_catalog_csv_adapter.py`
- `tests/unit/test_tabular_csv_adapter.py`
- `tests/unit/test_benchmark_label_service.py`
- `tests/unit/test_csv_pipeline.py`
- `tests/integration/test_csv_pipeline_cli.py`

**Modify**

- `app/schemas/benchmark.py`: nullable labels/scores, `UNLABELED`, and CSV run metadata.
- `app/repositories/benchmark_target.py`: persist/query CSV metadata and nullable labels.
- `app/services/benchmark_evaluator.py`: exclude unlabeled cases and return unavailable scores as `None`.
- `app/api/benchmark.py`: expose source metadata and `UNLABELED` filtering.
- `sql/benchmark_target_schema.sql`: fresh-install CSV columns.
- `README.md`: exact direct CSV commands and limits.
- Existing Benchmark tests: preserve current labeled MySQL behavior.

This phase does not delete `benchmark_import_service.py`, `benchmark_source.py`, or
`benchmark_field_input`; consolidation is a separate follow-up after the new path is proven.

## Task 1: Add canonical CSV input schemas and streaming reader

**Files:** create `app/schemas/csv_input.py`, `app/services/csv_reader.py`, and
`tests/unit/test_csv_reader.py`.

- [ ] Write failing tests using `tmp_path` for UTF-8 BOM, UTF-8, GB18030, empty files,
  duplicate/blank headers, 100 MB size checks through injected limits, row limits, and column limits.
- [ ] Run `python -m pytest tests/unit/test_csv_reader.py -q` and confirm missing-module failure.
- [ ] Implement these contracts:

```python
InputMode = Literal["auto", "catalog", "tabular"]
ResolvedInputMode = Literal["catalog", "tabular"]

class CSVReadLimits(BaseModel):
    max_bytes: int = 100 * 1024 * 1024
    max_rows: int = 1_000_000
    max_columns: int = 10_000

class CSVFieldCase(BaseModel):
    case_index: int = Field(ge=1)
    field_profile: FieldProfile
    expected_personal: bool | None = None

class CSVInputBatch(BaseModel):
    source_name: str
    source_fingerprint: str
    input_mode: ResolvedInputMode
    cases: list[CSVFieldCase] = Field(default_factory=list)
```

`CSVReader.inspect(path)` returns validated headers, encoding, safe basename, SHA-256, and row
count without retaining cell values. `CSVReader.iter_rows(path, inspection)` reopens the file and
yields `(row_number, dict[str, str])`. Encoding validation must scan incrementally with strict
decoders in order `utf-8-sig`, `utf-8`, `gb18030`.

- [ ] Re-run the focused test and `python -m ruff check app tests`.
- [ ] Commit `feat: add validated streaming csv reader`.

## Task 2: Detect modes and adapt both CSV shapes

**Files:** create `app/services/csv_mode_detector.py`, `catalog_csv_adapter.py`,
`tabular_csv_adapter.py`, and their three test files.

- [ ] Write failing detector tests: strong `字段名 + 样本1` and `field_name + sample_1` resolve to
  catalog; ordinary headers resolve to tabular; strong signature plus an unknown metadata header
  raises an ambiguity error; explicit mode bypasses detection but retains adapter validation.
- [ ] Write failing catalog tests: aliases map correctly, numeric sample ordering is stable, explicit
  `--field-name-column`/sample column mappings work, same names on different rows remain separate,
  and explicit catalog without samples yields an empty list.
- [ ] Write failing tabular tests: each header produces one case, the first five nonempty values are
  retained and truncated to 50 characters, empty columns remain cases, and no filename or label text
  enters `FieldProfile`.
- [ ] Run all three files and confirm expected missing implementations.
- [ ] Implement `resolve_csv_mode(headers, requested_mode)` with the exact recognized catalog
  metadata aliases from the design. Implement adapters with:

```python
CatalogCSVAdapter.load(path, *, input_mode, field_name_column=None,
                       sample_columns=None) -> CSVInputBatch
TabularCSVAdapter.load(path) -> CSVInputBatch
```

Both use `source_system="csv"`, `database_name="csv_source"`, and mode-specific neutral table
names. Neither accepts or reads labels.

- [ ] Run focused tests, all unit tests, and Ruff.
- [ ] Commit `feat: adapt catalog and tabular csv inputs`.

## Task 3: Parse and attach optional sidecar labels

**Files:** create `app/services/benchmark_label_service.py` and
`tests/unit/test_benchmark_label_service.py`; modify `app/schemas/csv_input.py`.

- [ ] Write failing tests for `true/false/1/0`, case-insensitive booleans, exact trimmed field-name
  matching, duplicate labels, invalid booleans, labels missing from the input, and source fields with
  no labels.
- [ ] Verify red with `python -m pytest tests/unit/test_benchmark_label_service.py -q`.
- [ ] Implement:

```python
class LabelMatchSummary(BaseModel):
    label_fingerprint: str
    labeled_cases: int
    unlabeled_cases: int
    cases: list[CSVFieldCase] = Field(default_factory=list)

def attach_benchmark_labels(
    batch: CSVInputBatch,
    label_path: str | Path,
) -> LabelMatchSummary: ...
```

The function validates the exact two-column header, never mutates `FieldProfile`, rejects extra label
fields, and returns copied cases with outer `expected_personal` values.

- [ ] Run focused tests and a leakage assertion that serialized `field_profile` contains neither
  `expected_personal` nor label-file information.
- [ ] Commit `feat: attach optional csv benchmark labels`.

## Task 4: Extend B models, metrics, and persistence compatibly

**Files:** modify `app/schemas/benchmark.py`, `benchmark_evaluator.py`,
`benchmark_target.py`, `sql/benchmark_target_schema.sql`; create the migration; extend existing unit
tests.

- [ ] Write failing tests proving labeled existing metrics are unchanged, unlabeled predictions are
  excluded, all score fields except Coverage are `None` when `labeled_cases=0`, and mixed input reports
  labeled/unlabeled counts.
- [ ] Write failing repository tests for CSV metadata, nullable `expected_personal`, outcome
  `UNLABELED`, source/label fingerprints, and query mapping.
- [ ] Run focused tests and confirm schema/model failures.
- [ ] Extend `BenchmarkRunSummary` with `source_type`, `input_mode`, `source_name`, fingerprints,
  `labeled_cases`, and `unlabeled_cases`. Extend outcome literals with `UNLABELED`; allow prediction
  labels to be null. Make unavailable score fields `float | None` while preserving numeric values for
  existing labeled runs.
- [ ] Update fresh SQL and add a clearly documented one-time migration that adds the run metadata,
  count columns, and makes `expected_personal` nullable. Do not drop or rename existing columns.
- [ ] Update target insert/update/read code and evaluator. Existing A-backed Benchmark tests must pass
  without changed expected scores.
- [ ] Run focused tests and the full suite.
- [ ] Commit `feat: persist direct csv classification runs`.

## Task 5: Implement direct classification and verified resume

**Files:** create `app/services/csv_pipeline.py`, `tests/unit/test_csv_pipeline.py`.

- [ ] Write failing orchestration tests for labeled TP/FP/TN/FN, unlabeled success, UNKNOWN failure,
  one failure continuing later cases, per-case persistence, and no label present in classifier inputs.
- [ ] Write failing resume tests: exact source and label fingerprints resume; changed source, changed
  labels, changed mode, or changed recorded `case_index + field_name` reject before classification;
  default skips all recorded cases and `retry_failed` retries only failures.
- [ ] Run the focused test and verify red.
- [ ] Implement `CSVClassificationPipeline.run(batch, label_summary=None)` and
  `resume(run_id, batch, label_summary=None, retry_failed=False)`. Reuse target checkpoint methods and
  `FieldClassificationService`; do not call the legacy A repository.
- [ ] Calculate `TP/FP/TN/FN` only when expected labels exist; otherwise save `UNLABELED`. Every
  successful/failed case is committed immediately, then metrics are rebuilt from B.
- [ ] Run focused and full unit suites and Ruff.
- [ ] Commit `feat: run direct csv classification pipeline`.

## Task 6: Add the CLI and read-only API compatibility

**Files:** create `scripts/run_csv_pipeline.py` and integration test; modify
`app/api/benchmark.py` and API tests.

- [ ] Write failing CLI parser tests for new run, resume, `auto|catalog|tabular`, optional labels,
  catalog mapping arguments, retry rules, and invalid argument combinations.
- [ ] Write failing API tests for CSV run metadata and `outcome=UNLABELED`.
- [ ] Implement new-run syntax:

```text
python -m scripts.run_csv_pipeline --input PATH
  [--input-mode auto|catalog|tabular] [--labels PATH]
  [--field-name-column NAME] [--sample-columns NAME1,NAME2]
```

and resume syntax:

```text
python -m scripts.run_csv_pipeline --resume-run UUID --input PATH
  [--labels PATH] [--retry-failed]
```

Construct the existing embedding/vector/LLM services lazily after all local CSV validation succeeds.
Print only source name, mode, run ID, counts, available scores, and status.

- [ ] Update API literals and response models without adding an HTTP run/upload endpoint.
- [ ] Run CLI help, CLI/API tests, full tests, and Ruff.
- [ ] Commit `feat: expose direct csv pipeline commands`.

## Task 7: Documentation, security checks, and final verification

**Files:** modify `README.md`; verify all new and modified files.

- [ ] Document catalog, tabular, auto detection, explicit override, label format, no-label semantics,
  one-file scope, B migration, resume fingerprints, cost warning, and exact commands.
- [ ] Add a no-paid-call integration smoke test using generated catalog/tabular/label files and fake
  classifier/target. Assert both shapes reach `FieldProfile`, labels stay external, and metrics match.
- [ ] Run fresh verification:

```powershell
python -m compileall app scripts
python -m ruff check .
python -m pytest -q -rs
python -m pip check
git diff --check
```

- [ ] Search runtime code for hard-coded teacher filenames, label leakage into prompt/classifier, CSV
  absolute paths in logs, and dead duplicate readers.
- [ ] Confirm `git status --short` contains no generated CSV, `.env`, cache, or runtime database.
- [ ] Commit `docs: explain direct csv input workflow`.
- [ ] Report exact verification counts and explicitly note that real MySQL/LLM/OCR and teacher files
  were not invoked.
