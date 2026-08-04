# Personal Information Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the two labeled CSV datasets into an A-database benchmark table, classify each stored row through the existing RAG service, persist predictions in B, and report leakage-safe personal-information detection scores.

**Architecture:** CSV parsing is a one-time import boundary; benchmark execution reads only MySQL A through a row-oriented repository. `FieldClassificationService` remains the single RAG + LLM decision path, while dedicated benchmark persistence and evaluation keep repeated field names and ground-truth labels separate from production field assets.

**Tech Stack:** Python 3.10, FastAPI, Pydantic, SQLAlchemy Core, MySQL 8, Chroma, LangChain, pytest, Ruff.

---

## File map

**Create**

- `app/schemas/benchmark.py`: benchmark cases, run summaries, predictions, metrics, and API response models.
- `app/repositories/benchmark_source.py`: transactional import and row-oriented reads from A.
- `app/repositories/benchmark_target.py`: benchmark run/prediction persistence and result queries in B.
- `app/services/benchmark_import_service.py`: encoding detection, CSV validation, label assignment, and sample normalization.
- `app/services/benchmark_evaluator.py`: pure confusion-matrix and metric calculation.
- `app/services/benchmark_pipeline.py`: sequential execution, checkpointing, resume, and retry-failed orchestration.
- `app/api/benchmark.py`: read-only benchmark run and result endpoints.
- `scripts/import_benchmark_data.py`: one-time CSV-to-A command.
- `scripts/run_benchmark.py`: local small-batch/full/resume runner.
- `sql/benchmark_source_schema.sql`: A benchmark input table.
- `sql/benchmark_target_schema.sql`: B benchmark run and prediction tables.
- `sql/migrations/2026-08-04_add_is_personal.sql`: one-time migration for an existing B database.
- `tests/unit/test_benchmark_import_service.py`
- `tests/unit/test_benchmark_evaluator.py`
- `tests/unit/test_benchmark_repositories.py`
- `tests/unit/test_benchmark_pipeline.py`
- `tests/integration/test_benchmark_api.py`

**Modify**

- `app/schemas/classification.py`: add explicit personal-information decision fields.
- `app/services/classification_service.py`: propagate `is_personal`; use `None` for `UNKNOWN`.
- `app/rag/prompt.py`: require an evidence-grounded personal-information decision.
- `app/repositories/target_mysql.py`: persist/query `is_personal` for the existing pipeline.
- `app/schemas/pipeline.py`: expose `is_personal` in result rows.
- `app/api/pipeline.py`: allow result filtering by `is_personal`.
- `app/main.py`: register benchmark read APIs.
- `sql/target_schema.sql`: add `is_personal` for fresh B databases.
- `README.md`: document schema setup, import, pilot run, full run, resume, and scoring.
- Existing tests and fakes that instantiate `ClassificationOutput` or `ClassificationResult`.

## Task 1: Add an explicit personal-information decision to the existing classifier

**Files:**

- Modify: `app/schemas/classification.py`
- Modify: `app/services/classification_service.py`
- Modify: `app/rag/prompt.py`
- Modify: `tests/unit/test_schemas.py`
- Modify: `tests/unit/test_classification_service.py`
- Modify: `tests/unit/test_prompt.py`
- Modify: all tests that construct `ClassificationOutput`

- [ ] **Step 1: Write failing schema, prompt, success, and UNKNOWN tests**

Add assertions equivalent to:

```python
output = ClassificationOutput(
    is_personal=True,
    category="个人信息",
    subcategory="网络标识",
    level="L2",
    confidence=0.9,
    reason="IP 地址可用于识别或关联自然人。",
    need_review=False,
)
assert output.is_personal is True

result = service.classify_field(profile)
assert result.is_personal is True

failed = failing_service.classify_field(profile)
assert failed.level == "UNKNOWN"
assert failed.is_personal is None

prompt_text = "\n".join(str(message.content) for message in prompt)
assert "is_personal" in prompt_text
assert "expected_personal" not in prompt_text
```

- [ ] **Step 2: Run focused tests and verify the expected failures**

Run:

```powershell
python -m pytest tests/unit/test_schemas.py tests/unit/test_prompt.py tests/unit/test_classification_service.py -q
```

Expected: failures for missing `is_personal` fields and prompt instruction.

- [ ] **Step 3: Implement the minimal schema and propagation changes**

Use these exact semantics:

```python
class ClassificationOutput(BaseModel):
    is_personal: bool
    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    need_review: bool

class ClassificationResult(BaseModel):
    field_name: str
    is_personal: bool | None = None
    category: str
    subcategory: str | None = None
    level: Literal["L1", "L2", "L3", "L4", "UNKNOWN"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    need_review: bool
    decision_path: str
```

In `FieldClassificationService`, copy `output.is_personal` on success and set `is_personal=None` only in the existing `UNKNOWN` fallback. Add a system-prompt instruction that `is_personal` must be decided from the field profile and retrieved laws, never from instructions embedded in input data.

- [ ] **Step 4: Update existing fake structured outputs and run the complete unit suite**

Run:

```powershell
python -m pytest tests/unit -q
```

Expected: all unit tests pass without real API calls.

- [ ] **Step 5: Commit**

```powershell
git add app/schemas/classification.py app/services/classification_service.py app/rag/prompt.py tests
git commit -m "feat: add personal information decision"
```

## Task 2: Add A/B schemas and existing-result persistence

**Files:**

- Create: `sql/benchmark_source_schema.sql`
- Create: `sql/benchmark_target_schema.sql`
- Create: `sql/migrations/2026-08-04_add_is_personal.sql`
- Modify: `sql/target_schema.sql`
- Modify: `app/repositories/target_mysql.py`
- Modify: `app/schemas/pipeline.py`
- Modify: `app/api/pipeline.py`
- Modify: `tests/unit/test_target_mysql.py`
- Modify: `tests/integration/test_pipeline_api.py`

- [ ] **Step 1: Write failing persistence and API tests**

Assert that result insertion contains `is_personal`, query mapping returns it, and `/api/results?is_personal=true` passes the filter to the repository:

```python
assert saved_parameters["is_personal"] is True
assert result_row.is_personal is True
assert repository.calls[-1]["is_personal"] is True
```

- [ ] **Step 2: Run focused tests and verify failures**

```powershell
python -m pytest tests/unit/test_target_mysql.py tests/integration/test_pipeline_api.py -q
```

Expected: failures because the column and filter do not exist.

- [ ] **Step 3: Add exact SQL structures**

Create `benchmark_field_input` in A with:

```sql
benchmark_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
batch_name VARCHAR(64) NOT NULL,
source_dataset VARCHAR(32) NOT NULL,
source_row_number INT NOT NULL,
field_name VARCHAR(128) NOT NULL,
sample_values_json JSON NOT NULL,
expected_personal BOOLEAN NOT NULL,
created_at DATETIME(6) NOT NULL,
UNIQUE KEY uq_benchmark_source_row
  (batch_name, source_dataset, source_row_number)
```

Create `benchmark_run` with this concrete shape:

```sql
CREATE TABLE IF NOT EXISTS benchmark_run (
  run_id CHAR(36) NOT NULL PRIMARY KEY,
  batch_name VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL,
  total_cases INT NOT NULL DEFAULT 0,
  success_cases INT NOT NULL DEFAULT 0,
  failed_cases INT NOT NULL DEFAULT 0,
  tp INT NOT NULL DEFAULT 0,
  fp INT NOT NULL DEFAULT 0,
  tn INT NOT NULL DEFAULT 0,
  fn INT NOT NULL DEFAULT 0,
  precision_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  recall_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  f1_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  accuracy_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  coverage_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  effective_recall_score DECIMAL(10,8) NOT NULL DEFAULT 0,
  model_name VARCHAR(128) NOT NULL,
  knowledge_base_version VARCHAR(64) NOT NULL,
  started_at DATETIME(6) NOT NULL,
  finished_at DATETIME(6) NULL,
  error_message TEXT NULL,
  KEY idx_benchmark_run_batch (batch_name),
  KEY idx_benchmark_run_started (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

Create `benchmark_prediction` with:

```sql
CREATE TABLE IF NOT EXISTS benchmark_prediction (
  prediction_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  run_id CHAR(36) NOT NULL,
  benchmark_id BIGINT NOT NULL,
  field_name_snapshot VARCHAR(128) NOT NULL,
  sample_values_json JSON NOT NULL,
  expected_personal BOOLEAN NOT NULL,
  predicted_personal BOOLEAN NULL,
  outcome VARCHAR(16) NOT NULL,
  category VARCHAR(256) NULL,
  subcategory VARCHAR(256) NULL,
  level VARCHAR(8) NULL,
  confidence DECIMAL(10,8) NULL,
  reason TEXT NULL,
  need_review BOOLEAN NULL,
  decision_path VARCHAR(128) NULL,
  evidence_json JSON NULL,
  status VARCHAR(16) NOT NULL,
  error_message TEXT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_benchmark_run_case (run_id, benchmark_id),
  KEY idx_benchmark_prediction_outcome (run_id, outcome),
  CONSTRAINT fk_benchmark_prediction_run
    FOREIGN KEY (run_id) REFERENCES benchmark_run (run_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

Add `is_personal BOOLEAN NOT NULL` to fresh `field_classification_result` definitions. The one-time migration must execute:

```sql
ALTER TABLE field_classification_result
  ADD COLUMN is_personal BOOLEAN NULL AFTER subcategory;
```

The nullable migration allows existing historical rows; all new successful rows must write a boolean.

- [ ] **Step 4: Update existing B persistence and queries**

Add `is_personal` to insert parameters, selected columns, `ClassificationResultRow`, and `/api/results` optional filters. Do not infer it from category or level.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest tests/unit/test_target_mysql.py tests/integration/test_pipeline_api.py -q
python -m pytest -q -rs
```

Expected: all ordinary tests pass; configured real integrations may remain explicitly skipped.

- [ ] **Step 6: Commit**

```powershell
git add sql app/repositories/target_mysql.py app/schemas/pipeline.py app/api/pipeline.py tests
git commit -m "feat: add benchmark database schemas"
```

## Task 3: Implement leakage-safe CSV import into A

**Files:**

- Create: `app/schemas/benchmark.py`
- Create: `app/services/benchmark_import_service.py`
- Create: `app/repositories/benchmark_source.py`
- Create: `scripts/import_benchmark_data.py`
- Create: `tests/unit/test_benchmark_import_service.py`
- Create: `tests/unit/test_benchmark_repositories.py`

- [ ] **Step 1: Write failing parser tests using generated temporary CSV files**

Tests must write tiny fixtures in `tmp_path`, one with `utf-8-sig` and one with `gb18030`, then assert:

```python
assert personal_case.expected_personal is True
assert non_personal_case.expected_personal is False
assert case.field_name == "reg_ip"
assert case.sample_values == ["192.168.*.*", "10.0.*.*"]
assert len(long_case.sample_values[0]) == 50
```

Also assert that duplicate field names on different source rows remain separate and malformed rows abort the combined batch before repository insertion.

- [ ] **Step 2: Run tests and verify missing-module failures**

```powershell
python -m pytest tests/unit/test_benchmark_import_service.py tests/unit/test_benchmark_repositories.py -q
```

- [ ] **Step 3: Implement parsing and validated import models**

Define models with these boundaries:

```python
class BenchmarkImportRow(BaseModel):
    source_dataset: Literal["personal", "non_personal"]
    source_row_number: int = Field(ge=2)
    field_name: str = Field(min_length=1, max_length=128)
    sample_values: list[str] = Field(default_factory=list, max_length=5)
    expected_personal: bool

class BenchmarkCase(BaseModel):
    benchmark_id: int
    batch_name: str
    expected_personal: bool
    field_profile: FieldProfile
```

Use `csv.DictReader`, encoding attempts in the approved order, exact header validation for `字段名`, ordered `样本N` columns, empty removal, first-five selection, and 50-character truncation. Never log sample content.

- [ ] **Step 4: Implement one-transaction import repository and CLI**

`BenchmarkSourceRepository.import_batch()` must accept both fully parsed lists and insert both datasets inside one `engine.begin()` block using:

```sql
INSERT INTO benchmark_field_input (
  batch_name, source_dataset, source_row_number, field_name,
  sample_values_json, expected_personal, created_at
) VALUES (
  :batch_name, :source_dataset, :source_row_number, :field_name,
  :sample_values_json, :expected_personal, :created_at
)
ON DUPLICATE KEY UPDATE benchmark_id = benchmark_id
```

Return inserted/skipped counts without deleting prior rows.

The CLI must require both file paths and `--batch`, load settings, parse both files before opening the write transaction, and print only counts.

- [ ] **Step 5: Run focused tests and CLI help**

```powershell
python -m pytest tests/unit/test_benchmark_import_service.py tests/unit/test_benchmark_repositories.py -q
python -m scripts.import_benchmark_data --help
```

Expected: tests pass and help lists `--personal`, `--non-personal`, and `--batch`.

- [ ] **Step 6: Commit**

```powershell
git add app/schemas/benchmark.py app/services/benchmark_import_service.py app/repositories/benchmark_source.py scripts/import_benchmark_data.py tests/unit/test_benchmark_import_service.py tests/unit/test_benchmark_repositories.py
git commit -m "feat: import benchmark fields into source database"
```

## Task 4: Implement deterministic benchmark metrics

**Files:**

- Create: `app/services/benchmark_evaluator.py`
- Modify: `app/schemas/benchmark.py`
- Create: `tests/unit/test_benchmark_evaluator.py`

- [ ] **Step 1: Write table-driven failing metric tests**

Cover all outcomes and zero denominators:

```python
metrics = evaluate_predictions(
    expected=[True, True, False, False],
    predicted=[True, False, True, False],
    failed=[False, False, False, False],
)
assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
assert metrics.precision_score == 0.5
assert metrics.recall_score == 0.5
assert metrics.f1_score == 0.5
assert metrics.coverage_score == 1.0
```

Add a failed positive case and assert `effective_recall_score` uses all selected real positives while ordinary recall uses successful cases only.

- [ ] **Step 2: Run test and verify failure**

```powershell
python -m pytest tests/unit/test_benchmark_evaluator.py -q
```

- [ ] **Step 3: Implement pure evaluation functions**

Use a `_safe_divide(numerator, denominator)` returning `0.0` for zero denominators. Do not query databases or call the classifier from this module. Return a Pydantic `BenchmarkMetrics` object whose counts and scores match the approved formulas.

- [ ] **Step 4: Run tests and commit**

```powershell
python -m pytest tests/unit/test_benchmark_evaluator.py -q
git add app/services/benchmark_evaluator.py app/schemas/benchmark.py tests/unit/test_benchmark_evaluator.py
git commit -m "feat: calculate benchmark detection metrics"
```

## Task 5: Implement checkpointed A-to-RAG-to-B benchmark execution

**Files:**

- Create: `app/repositories/benchmark_target.py`
- Create: `app/services/benchmark_pipeline.py`
- Create: `scripts/run_benchmark.py`
- Create: `tests/unit/test_benchmark_pipeline.py`
- Extend: `tests/unit/test_benchmark_repositories.py`

- [ ] **Step 1: Write failing orchestration tests**

Use fakes to assert:

```python
summary = pipeline.run(batch_name="teacher_2026_08", personal_limit=1, non_personal_limit=2)
assert summary.total_cases == 3
assert classifier.seen_field_names == ["email", "created_at", "price"]
assert target.saved_predictions == 3
```

Add tests that `UNKNOWN`/`is_personal=None` becomes `FAILED`, one failure does not stop later cases, resume skips recorded cases, and `retry_failed=True` retries only failed cases.

- [ ] **Step 2: Run tests and verify missing implementations**

```powershell
python -m pytest tests/unit/test_benchmark_pipeline.py tests/unit/test_benchmark_repositories.py -q
```

- [ ] **Step 3: Implement B repository**

Provide methods with stable contracts:

```python
create_run(summary, model_name, knowledge_version)
save_prediction(prediction)
list_recorded_case_ids(run_id, include_failed=True)
delete_failed_prediction(run_id, benchmark_id)
load_metric_inputs(run_id)
update_run(summary)
get_run(run_id)
query_predictions(
    run_id,
    outcome=None,
    predicted_personal=None,
    need_review=None,
    limit=100,
    offset=0,
)
```

Each prediction write uses its own transaction. Map SQLAlchemy errors to a benchmark persistence exception without exposing credentials.

- [ ] **Step 4: Implement pipeline and CLI**

The pipeline must select personal and non-personal limits independently, create one run, classify sequentially, persist every success/failure, aggregate metrics from B, and update the final run status. Resume must reuse the stored batch and version metadata.

CLI arguments must be mutually valid:

```text
new run: --batch [--personal-limit N] [--non-personal-limit N]
resume:  --resume-run UUID [--retry-failed]
```

Reject mixing `--batch` and `--resume-run`. Print run ID, counts, confusion matrix, precision, recall, F1, coverage, and effective recall without samples.

- [ ] **Step 5: Run focused tests and command help**

```powershell
python -m pytest tests/unit/test_benchmark_pipeline.py tests/unit/test_benchmark_repositories.py -q
python -m scripts.run_benchmark --help
```

- [ ] **Step 6: Commit**

```powershell
git add app/repositories/benchmark_target.py app/services/benchmark_pipeline.py scripts/run_benchmark.py app/schemas/benchmark.py tests/unit/test_benchmark_pipeline.py tests/unit/test_benchmark_repositories.py
git commit -m "feat: run checkpointed personal information benchmark"
```

## Task 6: Add read-only Benchmark APIs

**Files:**

- Create: `app/api/benchmark.py`
- Modify: `app/main.py`
- Create: `tests/integration/test_benchmark_api.py`

- [ ] **Step 1: Write failing endpoint tests**

Using dependency overrides, assert:

```python
response = client.get(f"/api/benchmark/runs/{run_id}")
assert response.status_code == 200
assert response.json()["f1_score"] == 0.8

response = client.get(
    "/api/benchmark/results",
    params={"run_id": str(run_id), "outcome": "FN", "limit": 20},
)
assert response.status_code == 200
assert all(row["outcome"] == "FN" for row in response.json())
```

Also test 404, invalid outcome, limit bounds, and missing `TARGET_DATABASE_URL`.

- [ ] **Step 2: Run test and verify 404/missing module failures**

```powershell
python -m pytest tests/integration/test_benchmark_api.py -q
```

- [ ] **Step 3: Implement router and register it**

Add:

```text
GET /api/benchmark/runs/{run_id}
GET /api/benchmark/results
```

Use lazy repository construction consistent with `app/api/pipeline.py`. API code may validate query parameters and map not-found results, but must not calculate metrics or write SQL.

- [ ] **Step 4: Run API and full tests, then commit**

```powershell
python -m pytest tests/integration/test_benchmark_api.py -q
python -m pytest -q -rs
git add app/api/benchmark.py app/main.py tests/integration/test_benchmark_api.py
git commit -m "feat: expose benchmark result queries"
```

## Task 7: Document, migrate, and verify the complete feature

**Files:**

- Modify: `README.md`
- Modify: `.gitignore` only if external Benchmark artifacts are not already covered.
- Verify all files from Tasks 1-6.

- [ ] **Step 1: Update README with exact commands**

Document this order:

```powershell
cmd /c "mysql -u root -p < sql\benchmark_source_schema.sql"
cmd /c "mysql -u root -p < sql\benchmark_target_schema.sql"
cmd /c "mysql -u root -p compliance_result < sql\migrations\2026-08-04_add_is_personal.sql"
python -m scripts.import_benchmark_data --personal "D:\benchmark\个人信息_脱敏.csv" --non-personal "D:\benchmark\非个人信息字段_脱敏.csv" --batch teacher_2026_08
python -m scripts.run_benchmark --batch teacher_2026_08 --personal-limit 20 --non-personal-limit 80
python -m scripts.run_benchmark --batch teacher_2026_08
python -m scripts.run_benchmark --resume-run <run_id> --retry-failed
```

Explain class imbalance, leakage prevention, metrics, and that CSV files remain outside Git.

- [ ] **Step 2: Run fresh complete verification**

```powershell
python -m compileall app scripts
python -m ruff check .
python -m pytest -q -rs
python -m pip check
git diff --check
```

Expected: compile and Ruff exit 0; all ordinary tests pass; only explicitly unconfigured real MySQL/Qwen/paid-LLM integrations may skip; no broken requirements or whitespace errors.

- [ ] **Step 3: Perform a no-paid-call local smoke test**

Use generated temporary CSV fixtures plus fake classifier/repositories to run import → classification → metric aggregation without teacher data or external APIs. Verify one TP, one FP, one TN, one FN and exact 0.5 precision/recall/F1.

- [ ] **Step 4: Inspect repository hygiene**

```powershell
git status --short --ignored
git grep -n "expected_personal" -- app/rag app/services/classification_service.py
git grep -n "个人信息_脱敏.csv\|非个人信息字段_脱敏.csv" -- . ":(exclude)docs/superpowers"
```

Expected: labels do not enter classifier/prompt code; teacher file names do not appear in runtime code; `.env`, CSV, caches, and runtime outputs remain ignored.

- [ ] **Step 5: Commit final documentation**

```powershell
git add README.md .gitignore
git commit -m "docs: explain benchmark evaluation workflow"
```

- [ ] **Step 6: Final report**

Report commits, schema changes, commands, exact test counts, skipped real integrations and reasons, and the commands the user must run with the real CSV files. Do not run the full paid 2357-case DeepSeek benchmark unless the user separately authorizes that cost.
