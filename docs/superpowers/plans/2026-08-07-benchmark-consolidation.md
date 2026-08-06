# Benchmark Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dedicated Benchmark CSV import, database-A staging, and classification pipeline with a thin two-file assembler over the generic catalog CSV adapter and CSV classification pipeline.

**Architecture:** Both teacher catalog CSV files are parsed by `CatalogCSVAdapter` with explicit Benchmark-compatible `FieldProfile` defaults. `benchmark_label_service` assigns labels from file roles, preserves deterministic ordering, computes a composite fingerprint, and returns the generic `CSVInputBatch` and `LabelMatchSummary`; `CSVClassificationPipeline` remains the only classifier/checkpoint pipeline and the B repository/API remain unchanged.

**Tech Stack:** Python 3.10+, Pydantic, pytest, FastAPI, SQLAlchemy, Ruff.

---

### Task 1: Add configurable catalog profile defaults

**Files:**
- Modify: `app/schemas/csv_input.py`
- Modify: `app/services/catalog_csv_adapter.py`
- Test: `tests/unit/test_catalog_csv_adapter.py`

- [ ] Add a failing test that calls `CatalogCSVAdapter().load(path, profile_defaults=CSVProfileDefaults(...))` and asserts all fallback metadata uses the supplied values while explicit CSV metadata still takes priority.
- [ ] Run `python -m pytest tests/unit/test_catalog_csv_adapter.py -q` and confirm failure because `CSVProfileDefaults` or `profile_defaults` does not exist.
- [ ] Add `CSVProfileDefaults` with defaults `csv`, `csv_source`, `catalog_input`, `unknown`, and `general`; accept it in `CatalogCSVAdapter.load` and use it only when a row lacks explicit metadata.
- [ ] Re-run the focused test and the existing catalog adapter tests.
- [ ] Commit as `feat: configure catalog csv profile defaults`.

### Task 2: Assemble the labeled two-file Benchmark through generic CSV models

**Files:**
- Modify: `app/services/benchmark_label_service.py`
- Test: `tests/unit/test_benchmark_label_service.py`

- [ ] Add failing tests for `prepare_labeled_catalog_benchmark`: true/false labels from file roles, personal-then-non-personal order, continuous case indexes, independent limits, duplicate field names, legacy `FieldProfile` metadata, label isolation, and a deterministic 64-character composite fingerprint that changes when file roles or limits change.
- [ ] Run the new tests and confirm they fail because the assembler does not exist.
- [ ] Implement `BenchmarkCSVInput` and `prepare_labeled_catalog_benchmark` using `CatalogCSVAdapter`, `CSVProfileDefaults`, SHA-256 over a versioned canonical payload, and generic `CSVInputBatch`/`LabelMatchSummary` results.
- [ ] Re-run `tests/unit/test_benchmark_label_service.py` and `tests/unit/test_catalog_csv_adapter.py`.
- [ ] Commit as `feat: assemble labeled benchmark csv inputs`.

### Task 3: Switch the Benchmark CLI to the generic CSV pipeline

**Files:**
- Modify: `scripts/run_benchmark.py`
- Modify: `app/services/csv_pipeline.py`
- Create/Rewrite: `tests/integration/test_benchmark_smoke.py`
- Create: `tests/integration/test_benchmark_cli.py`

- [ ] Add failing CLI tests requiring `--personal` and `--non-personal`, rejecting retry without resume, and rejecting limits during resume.
- [ ] Add a failing offline smoke test that parses two generated files, runs `CSVClassificationPipeline` with a fake classifier, and obtains TP=1, FP=1, TN=1, FN=1.
- [ ] Add a failing pipeline test proving the run records the supplied batch name and per-class limits without changing generic CSV behavior.
- [ ] Implement the thin CLI: load both files through `prepare_labeled_catalog_benchmark`, build `CSVClassificationPipeline`, and call `run` or `resume` with the same batch and labels.
- [ ] Extend the generic pipeline run metadata through optional batch metadata on `CSVInputBatch`, keeping existing one-file defaults backward compatible.
- [ ] Run the focused CLI, smoke, and CSV pipeline tests.
- [ ] Commit as `refactor: run benchmark through csv pipeline`.

### Task 4: Remove obsolete A-staging and duplicate pipeline code

**Files:**
- Delete: `app/services/benchmark_import_service.py`
- Delete: `app/services/benchmark_pipeline.py`
- Delete: `app/repositories/benchmark_source.py`
- Delete: `scripts/import_benchmark_data.py`
- Delete: `sql/benchmark_source_schema.sql`
- Modify: `app/schemas/benchmark.py`
- Delete: `tests/unit/test_benchmark_import_service.py`
- Delete: `tests/unit/test_benchmark_pipeline.py`
- Modify: `tests/unit/test_benchmark_repositories.py`
- Modify: `tests/unit/test_sql_assets.py`

- [ ] Search tracked source and tests for every obsolete module/model reference and record the remaining callers.
- [ ] Remove tests tied only to A staging; keep target repository coverage and historical `mysql_benchmark` schema compatibility coverage.
- [ ] Delete the obsolete modules and remove `DatasetLabel`, `BenchmarkImportRow`, `BenchmarkImportSummary`, and `BenchmarkCase`.
- [ ] Update SQL asset tests to require only the Benchmark target schema.
- [ ] Run all Benchmark, CSV, repository, schema, and API tests.
- [ ] Commit as `refactor: remove benchmark source staging`.

### Task 5: Update operational documentation

**Files:**
- Modify: `README.md`
- Delete: `docs/superpowers/plans/2026-08-04-benchmark-test.md` if present and obsolete

- [ ] Replace the two-step import/run instructions with direct two-file pilot, full, resume, and result-query commands.
- [ ] State that Benchmark needs only the B target schema, while `SOURCE_DATABASE_URL` remains for the ordinary MySQL A pipeline.
- [ ] Remove references to the deleted source table, import script, SQL file, and dedicated pipeline.
- [ ] Search README and active source for obsolete names and confirm no runtime references remain.
- [ ] Commit as `docs: update benchmark csv workflow`.

### Task 6: Full verification and completion review

**Files:**
- Verify all changed files

- [ ] Run `python -m compileall app scripts`.
- [ ] Run `ruff check .`.
- [ ] Run `pytest -q`.
- [ ] Run searches for `benchmark_import_service`, `benchmark_source`, `benchmark_pipeline`, `import_benchmark_data`, `benchmark_source_schema`, `BenchmarkImportRow`, and `BenchmarkCase`; only historical design documents may mention them.
- [ ] Run `git diff --check`, inspect `git status`, and inspect the complete branch diff.
- [ ] Confirm no generated runtime files are tracked and no database table was dropped.
- [ ] Use the finishing-a-development-branch workflow to report integration options without pushing.
