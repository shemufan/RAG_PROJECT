# Database Classification Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the synchronous `enterprise_source` MySQL → existing RAG classifier → `compliance_result` MySQL demo workflow with query APIs and executable SQL assets.

**Architecture:** Use SQLAlchemy 2.x Core repositories for two independent MySQL connections, strict Pydantic boundary models, and a synchronous orchestration service. Keep SQL in repositories, RAG in `FieldClassificationService`, and HTTP adaptation in FastAPI routes; instantiate database dependencies lazily so the existing application can start without MySQL.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic 2, SQLAlchemy 2.x Core, PyMySQL, Chroma/LangChain, pytest, httpx, Ruff.

---

## File Map

- Modify `app/schemas/field.py`: expanded, backward-compatible `FieldProfile` validation.
- Create `app/schemas/pipeline.py`: run, record, result, and evidence models plus stable field IDs.
- Create `app/repositories/source_mysql.py`: source metadata scan, safe sampling, mapping, masking.
- Create `app/repositories/target_mysql.py`: target writes and relational result queries.
- Create `app/services/database_pipeline.py`: run orchestration and status accounting.
- Create `app/api/pipeline.py`: four pipeline and query endpoints with lazy dependencies.
- Modify `app/main.py`: register the new router only; do not connect to MySQL at startup.
- Modify `app/core/config.py`, `.env.example`: add two independent database URLs.
- Create `sql/source_schema.sql`, `sql/source_seed.sql`, `sql/target_schema.sql`, `sql/query_examples.sql`.
- Create five requested unit/integration test modules and extend existing regression tests.
- Modify `requirements.txt`, `README.md`, and schema package exports.

### Task 1: Expanded Field Profile and Database Configuration

**Files:**
- Modify: `tests/unit/test_schemas.py`
- Modify: `tests/unit/test_config.py`
- Modify: `app/schemas/field.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing schema and config tests**

Add tests that preserve the manual API defaults while enforcing database-scanned values:

```python
def test_field_profile_has_pipeline_metadata_and_manual_defaults():
    profile = FieldProfile(field_name="employee_id")
    assert profile.schema_version == "1.0"
    assert profile.source_system == "manual"
    assert profile.database_name == "manual"
    assert profile.table_name == "manual"
    assert profile.data_type == "unknown"
    assert profile.is_nullable is True


def test_field_profile_rejects_empty_identity_and_long_samples():
    for field, value in (
        ("field_name", ""),
        ("database_name", ""),
        ("table_name", ""),
    ):
        payload = {"field_name": "name", field: value}
        with pytest.raises(ValidationError):
            FieldProfile(**payload)
    with pytest.raises(ValidationError):
        FieldProfile(field_name="name", sample_values=["x" * 51])


def test_config_exposes_independent_database_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATABASE_URL", "mysql+pymysql://source/db")
    monkeypatch.setenv("TARGET_DATABASE_URL", "mysql+pymysql://target/db")
    settings = load_settings(tmp_path)
    assert settings.source_database_url.endswith("/db")
    assert settings.target_database_url.endswith("/db")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_schemas.py tests\unit\test_config.py
```

Expected: failures for missing `schema_version`, database metadata, and URL settings.

- [ ] **Step 3: Implement the minimal validated schema and settings**

Use constrained strings and defaults that keep manual `/api/classify` requests valid:

```python
class FieldProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_system: RequiredName = "manual"
    database_name: RequiredName = "manual"
    table_name: RequiredName = "manual"
    table_comment: LongText | None = None
    field_name: FieldName
    field_cn: ShortText | None = None
    field_comment: LongText | None = None
    data_type: RequiredName = "unknown"
    is_nullable: bool = True
    column_key: ShortText | None = None
    sample_values: list[SampleValue] = Field(default_factory=list, max_length=5)
    business_domain: ShortText = "general"
```

Add nullable settings fields populated only from the root environment:

```python
source_database_url: str | None
target_database_url: str | None

source_database_url=os.getenv("SOURCE_DATABASE_URL") or None,
target_database_url=os.getenv("TARGET_DATABASE_URL") or None,
```

Append to `.env.example`:

```dotenv
SOURCE_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/enterprise_source
TARGET_DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/compliance_result
```

Append runtime dependencies:

```text
sqlalchemy>=2.0,<3.0
pymysql>=1.1,<2.0
```

- [ ] **Step 4: Install dependencies and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_schemas.py tests\unit\test_config.py
```

Expected: all schema and config tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add app/schemas/field.py app/core/config.py tests/unit/test_schemas.py tests/unit/test_config.py .env.example requirements.txt
git commit -m "feat: extend field profile for database metadata"
```

### Task 2: Pipeline Schemas and Stable Field Identity

**Files:**
- Create: `tests/unit/test_target_record_mapping.py`
- Create: `app/schemas/pipeline.py`
- Modify: `app/schemas/__init__.py`

- [ ] **Step 1: Write failing stable-ID and record tests**

```python
from uuid import uuid4

from app.schemas.classification import ClassificationOutput, Evidence
from app.schemas.field import FieldProfile
from app.schemas.pipeline import FieldClassificationRecord, stable_field_id


def make_profile() -> FieldProfile:
    return FieldProfile(
        source_system="mysql",
        database_name="enterprise_source",
        table_name="employee",
        field_name="id_card_no",
        data_type="varchar(18)",
        is_nullable=False,
        business_domain="hr",
    )


def test_stable_field_id_is_repeatable_and_field_specific():
    profile = make_profile()
    assert stable_field_id(profile) == stable_field_id(profile)
    assert stable_field_id(profile) != stable_field_id(
        profile.model_copy(update={"field_name": "phone"})
    )


def test_field_classification_record_keeps_input_output_and_evidence():
    profile = make_profile()
    record = FieldClassificationRecord(
        run_id=uuid4(),
        field_id=stable_field_id(profile),
        field_profile=profile,
        classification=ClassificationOutput(
            category="个人信息",
            subcategory="身份标识",
            level="L4",
            confidence=0.95,
            reason="法规明确规定",
            need_review=False,
        ),
        evidence=[Evidence(source="个人信息保护法.txt", content="法规内容", score=0.9)],
        decision_path="rag_llm",
        model_name="deepseek-chat",
        knowledge_base_version="v1",
    )
    assert record.field_id == stable_field_id(profile)
    assert record.evidence[0].source == "个人信息保护法.txt"
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_target_record_mapping.py
```

Expected: import failure because `app.schemas.pipeline` does not exist.

- [ ] **Step 3: Implement strict pipeline models**

Create these public types with Pydantic validation:

```python
RunStatus = Literal["RUNNING", "SUCCESS", "PARTIAL_FAILED", "FAILED"]


def stable_field_id(profile: FieldProfile) -> UUID:
    identity = ":".join(
        (
            profile.source_system,
            profile.database_name,
            profile.table_name,
            profile.field_name,
        )
    )
    return uuid5(NAMESPACE_URL, identity)


class FieldClassificationRecord(BaseModel):
    run_id: UUID
    field_id: UUID
    field_profile: FieldProfile
    classification: ClassificationOutput
    evidence: list[Evidence] = Field(default_factory=list)
    decision_path: str
    model_name: str
    knowledge_base_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineRequest(BaseModel):
    sample_limit: int = Field(default=3, ge=0, le=5)
    table_names: list[str] | None = None
    continue_on_error: bool = True


class PipelineSummary(BaseModel):
    run_id: UUID
    source_database: str
    total_fields: int = Field(ge=0)
    success_fields: int = Field(ge=0)
    review_fields: int = Field(ge=0)
    failed_fields: int = Field(ge=0)
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
```

Also define `RunDetail`, `ClassificationResultRow`, and `ClassificationEvidenceRow` with fields
matching the target SQL tables and API responses exactly.

- [ ] **Step 4: Run and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_target_record_mapping.py tests\unit\test_schemas.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add app/schemas/pipeline.py app/schemas/__init__.py tests/unit/test_target_record_mapping.py
git commit -m "feat: add pipeline records and stable field identity"
```

### Task 3: Source MySQL Metadata Mapping and Masked Sampling

**Files:**
- Create: `tests/unit/test_source_field_mapping.py`
- Create: `app/repositories/source_mysql.py`
- Modify: `app/repositories/__init__.py`

- [ ] **Step 1: Write failing pure mapping and masking tests**

```python
import pytest

from app.repositories.source_mysql import (
    business_domain_for_table,
    map_metadata_row,
    mask_sample_value,
    validate_sample_limit,
)


METADATA_ROW = {
    "TABLE_SCHEMA": "enterprise_source",
    "TABLE_NAME": "employee",
    "TABLE_COMMENT": "员工信息",
    "COLUMN_NAME": "phone",
    "COLUMN_TYPE": "varchar(20)",
    "COLUMN_COMMENT": "联系电话",
    "IS_NULLABLE": "NO",
    "COLUMN_KEY": "",
    "ORDINAL_POSITION": 4,
}


def test_information_schema_row_maps_to_field_profile():
    profile = map_metadata_row(METADATA_ROW, ["13812345678"])
    assert profile.database_name == "enterprise_source"
    assert profile.table_name == "employee"
    assert profile.is_nullable is False
    assert profile.business_domain == "hr"
    assert profile.sample_values == ["138****5678"]


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("id_card_no", "110101199001011234", "1101**********1234"),
        ("mobile", "13812345678", "138****5678"),
        ("bank_card_no", "6222021234567890123", "6222***********0123"),
        ("email", "alice@example.test", "a****@example.test"),
    ],
)
def test_sample_masking(name, value, expected):
    assert mask_sample_value(name, value) == expected


def test_sample_limit_boundaries():
    assert validate_sample_limit(0) == 0
    assert validate_sample_limit(5) == 5
    with pytest.raises(ValueError):
        validate_sample_limit(6)


def test_business_domain_mapping_is_explicit():
    assert business_domain_for_table("customer_order") == "commerce"
    assert business_domain_for_table("unknown") == "general"
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_source_field_mapping.py
```

Expected: import failure because `source_mysql.py` does not exist.

- [ ] **Step 3: Implement mapping, masking, and read-only scan**

Create an explicit domain map, masking helpers, and a repository whose public method is:

```python
class SourceMySQLRepository:
    def __init__(self, database_url: str | None = None, *, engine=None):
        if engine is None:
            if not database_url:
                raise ValueError("SOURCE_DATABASE_URL 未配置")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine
        self.database_name = engine.url.database
        if not self.database_name:
            raise ValueError("SOURCE_DATABASE_URL 必须包含数据库名")

    def scan_fields(
        self,
        sample_limit: int = 3,
        table_names: list[str] | None = None,
    ) -> list[FieldProfile]:
        limit = validate_sample_limit(sample_limit)
        metadata = self._load_metadata(table_names)
        profiles = []
        for row in metadata:
            samples = self._load_samples(row, limit)
            profiles.append(map_metadata_row(row, samples))
        return profiles
```

Use one parameterized metadata query joining `information_schema.COLUMNS AS c` to
`information_schema.TABLES AS t`, ordered by table and ordinal position. If `table_names` is
provided, use SQLAlchemy `bindparam("table_names", expanding=True)`. Reject names absent from the
metadata result by never accepting them into `_load_samples`.

For sample SQL, quote the table and column through `self.engine.dialect.identifier_preparer.quote`
and use only the returned metadata values:

```python
statement = text(
    f"SELECT DISTINCT {quoted_column} AS sample_value "
    f"FROM {quoted_table} WHERE {quoted_column} IS NOT NULL LIMIT :sample_limit"
)
```

Catch only sample-query `SQLAlchemyError`, log the table/column identity without values, and
return `[]`. Do not catch metadata query errors. Deduplicate masked values in insertion order and
cap every final sample to fifty characters.

- [ ] **Step 4: Add a fake-engine scan test and verify GREEN**

Add a fake connection that returns one metadata row and sample row, then assert
`scan_fields(sample_limit=3)` returns one fully mapped profile and executes no write statement.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_source_field_mapping.py
```

Expected: all source mapping, masking, boundary, and scan tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add app/repositories/source_mysql.py app/repositories/__init__.py tests/unit/test_source_field_mapping.py
git commit -m "feat: scan and mask source mysql fields"
```

### Task 4: Target MySQL Persistence and Queries

**Files:**
- Extend: `tests/unit/test_target_record_mapping.py`
- Create: `app/repositories/target_mysql.py`
- Modify: `app/repositories/__init__.py`

- [ ] **Step 1: Write failing target-parameter and transaction tests**

Use a fake engine whose `begin()` context records SQL and parameters:

```python
def test_target_repository_maps_asset_and_record_to_relational_columns(record, fake_engine):
    repository = TargetMySQLRepository(engine=fake_engine)
    repository.upsert_field_asset(record.field_id, record.field_profile, record.created_at)
    result_id = repository.save_classification_record(record)

    assert result_id == 41
    asset = fake_engine.parameters_for("INSERT INTO data_field_asset")
    assert asset["column_name"] == "id_card_no"
    assert asset["database_name"] == "enterprise_source"
    result = fake_engine.parameters_for("INSERT INTO field_classification_result")
    assert result["level"] == "L4"
    assert json.loads(result["input_snapshot_json"])["schema_version"] == "1.0"
    evidence = fake_engine.parameters_for("INSERT INTO classification_evidence")
    assert evidence[0]["rank_no"] == 1


def test_save_record_rolls_back_and_raises_context_on_evidence_failure(record, failing_engine):
    repository = TargetMySQLRepository(engine=failing_engine)
    with pytest.raises(TargetPersistenceError, match="save classification record"):
        repository.save_classification_record(record)
    assert failing_engine.rolled_back is True
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_target_record_mapping.py
```

Expected: import failure for `TargetMySQLRepository`.

- [ ] **Step 3: Implement explicit SQLAlchemy Core operations**

Create `TargetPersistenceError(RuntimeError)` and implement these exact public calls:

- `TargetMySQLRepository(database_url: str | None = None, *, engine=None)`
- `create_run(summary: PipelineSummary, model_name: str, knowledge_version: str) -> None`
- `update_run(summary: PipelineSummary, error_message: str | None = None) -> None`
- `upsert_field_asset(field_id: UUID, profile: FieldProfile, seen_at: datetime) -> None`
- `save_classification_record(record: FieldClassificationRecord) -> int`
- `get_run(run_id: UUID) -> RunDetail | None`
- `query_results(*, run_id: UUID | None = None, database_name: str | None = None,
  table_name: str | None = None, column_name: str | None = None,
  level: str | None = None, category: str | None = None,
  need_review: bool | None = None, limit: int = 100,
  offset: int = 0) -> list[ClassificationResultRow]`
- `get_result_evidence(result_id: int) -> list[ClassificationEvidenceRow]`

Use MySQL's `ON DUPLICATE KEY UPDATE` clause for field assets. Serialize snapshots with
`model_dump(mode="json")`, `ensure_ascii=False`, and stable separators. `save_classification_record`
must execute its result insert, obtain `lastrowid`, and insert evidence rows in the same
`engine.begin()` context. Wrap `SQLAlchemyError` as `TargetPersistenceError` with operation name
and preserve the original exception as `__cause__`.

Build `query_results` from a fixed base statement and a fixed allowlist of optional predicates;
all values remain bound parameters. Order by result creation time and ID descending. Order
evidence by rank ascending.

- [ ] **Step 4: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_target_record_mapping.py
```

Expected: all target mapping, transaction, query-filter, and evidence-order tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add app/repositories/target_mysql.py app/repositories/__init__.py tests/unit/test_target_record_mapping.py
git commit -m "feat: persist and query compliance results"
```

### Task 5: Database Classification Pipeline

**Files:**
- Create: `tests/unit/test_database_pipeline.py`
- Create: `app/services/database_pipeline.py`
- Modify: `app/services/__init__.py`

- [ ] **Step 1: Write failing happy-path and identity tests**

Create fake source, target, and classifier objects that store every call. Test two invocations:

```python
def test_pipeline_success_uses_new_run_and_stable_field_ids(dependencies):
    first = dependencies.pipeline.run(PipelineRequest())
    second = dependencies.pipeline.run(PipelineRequest())
    assert first.run_id != second.run_id
    assert dependencies.saved_records[0].field_id == dependencies.saved_records[1].field_id
    assert first.status == "SUCCESS"
    assert first.total_fields == first.success_fields == 1
    assert first.failed_fields == 0
```

- [ ] **Step 2: Write failing review and failure-policy tests**

```python
def test_pipeline_counts_review_without_failure(review_dependencies):
    summary = review_dependencies.pipeline.run(PipelineRequest())
    assert summary.status == "SUCCESS"
    assert summary.review_fields == 1
    assert summary.failed_fields == 0


def test_pipeline_continues_after_unknown_and_does_not_save_failed_result(mixed_dependencies):
    summary = mixed_dependencies.pipeline.run(PipelineRequest(continue_on_error=True))
    assert summary.status == "PARTIAL_FAILED"
    assert summary.success_fields == 1
    assert summary.failed_fields == 1
    assert len(mixed_dependencies.saved_records) == 1
    assert len(mixed_dependencies.upserted_assets) == 2


def test_pipeline_all_unknown_is_failed(all_failed_dependencies):
    summary = all_failed_dependencies.pipeline.run(PipelineRequest())
    assert summary.status == "FAILED"
    assert summary.success_fields == 0
    assert summary.failed_fields == summary.total_fields
```

Also test that `continue_on_error=False` stops after the first failure and that a source scan
exception finalizes the run as `FAILED` with zero fields.

- [ ] **Step 3: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_database_pipeline.py
```

Expected: import failure because `database_pipeline.py` does not exist.

- [ ] **Step 4: Implement orchestration without SQL**

Create `DatabaseClassificationPipeline` with constructor dependencies `source_repository`,
`target_repository`, and `classification_service`, plus keyword-only `model_name`,
`knowledge_base_version`, injectable UTC `clock`, and injectable UUID4 `run_id_factory`.
Its public method is `run(request: PipelineRequest) -> PipelineSummary`.

The implementation must:

1. Construct and persist a `RUNNING` summary before scanning.
2. Scan using only `sample_limit` and `table_names` from the request.
3. Upsert each field asset before classification persistence.
4. Convert successful `ClassificationResult` into a strict `ClassificationOutput`.
5. Construct and save `FieldClassificationRecord` with evidence order unchanged.
6. Treat `UNKNOWN`, classifier exceptions, or target persistence exceptions as field failures.
7. Continue or stop according to the request.
8. Derive status through one small `_final_status(total, success, failed)` function.
9. Finalize the run and return its validated summary.

Do not catch `create_run` or final `update_run` persistence failures. Log field identities but no
sample values or database credentials.

- [ ] **Step 5: Verify GREEN and full fake chain**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_database_pipeline.py
```

Expected: all success, review, partial failure, total failure, stop policy, scan failure, ID, and
evidence-order tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add app/services/database_pipeline.py app/services/__init__.py tests/unit/test_database_pipeline.py
git commit -m "feat: orchestrate database classification runs"
```

### Task 6: Pipeline FastAPI Routes and Lazy Dependencies

**Files:**
- Create: `tests/integration/test_pipeline_api.py`
- Create: `app/api/pipeline.py`
- Modify: `app/api/__init__.py`
- Modify: `app/main.py`
- Extend: `tests/integration/test_api.py`

- [ ] **Step 1: Write failing API tests with dependency overrides**

Create an application using the existing empty lifespan and override `get_database_pipeline` and
`get_target_repository`. Test:

```python
def test_pipeline_run_endpoint_returns_summary(client):
    response = client.post(
        "/api/pipeline/run",
        json={"sample_limit": 3, "table_names": None, "continue_on_error": True},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"


def test_run_endpoint_returns_404_for_missing_run(client):
    response = client.get(f"/api/runs/{uuid4()}")
    assert response.status_code == 404


def test_results_query_validates_limit(client):
    assert client.get("/api/results?limit=201").status_code == 422


def test_result_evidence_endpoint_returns_ranked_rows(client):
    response = client.get("/api/results/41/evidence")
    assert response.status_code == 200
    assert response.json()[0]["rank_no"] == 1
```

Extend the original classification test to continue posting a minimal manual profile and assert
HTTP 200.

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\integration\test_pipeline_api.py tests\integration\test_api.py
```

Expected: new route requests return 404 because the router does not exist.

- [ ] **Step 3: Implement routes and lazy dependencies**

Create synchronous routes with response models:

```python
@router.post("/pipeline/run", response_model=PipelineSummary)
def run_pipeline(request: PipelineRequest, pipeline=Depends(get_database_pipeline)):
    return pipeline.run(request)


@router.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: UUID, repository=Depends(get_target_repository)):
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="classification run not found")
    return run
```

Add `/results` with typed optional filters, `limit: int = Query(100, ge=1, le=200)`, and
`offset: int = Query(0, ge=0)`. Add `/results/{result_id}/evidence` with `result_id >= 1`.

`get_target_repository` and `get_database_pipeline` must read settings server-side and raise HTTP
503 when the relevant URL is absent. Cache created repositories in `request.app.state` only after
the endpoint is called. Reuse `request.app.state.classification_service`; do not construct a
second Chroma or LLM service.

Register the router in `create_app` without changing `app_lifespan`.

- [ ] **Step 4: Verify GREEN and OpenAPI registration**

Add an assertion that `application.openapi()["paths"]` includes all four new paths and
`/api/classify`.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\integration\test_pipeline_api.py tests\integration\test_api.py
```

Expected: all API tests pass without MySQL, LLM, embedding model, or started service.

- [ ] **Step 5: Commit Task 6**

```powershell
git add app/api/pipeline.py app/api/__init__.py app/main.py tests/integration/test_pipeline_api.py tests/integration/test_api.py
git commit -m "feat: expose database pipeline api"
```

### Task 7: Executable Source, Target, Seed, and Query SQL

**Files:**
- Create: `tests/unit/test_sql_assets.py`
- Create: `sql/source_schema.sql`
- Create: `sql/source_seed.sql`
- Create: `sql/target_schema.sql`
- Create: `sql/query_examples.sql`

- [ ] **Step 1: Write failing static SQL consistency tests**

```python
def test_source_schema_has_required_business_tables_and_no_answers():
    sql = read_sql("source_schema.sql").lower()
    for table in ("employee", "customer_account", "customer_order", "product"):
        assert f"create table {table}" in normalized(sql)
    for forbidden in ("sensitivity", "classification_level", "category_level"):
        assert forbidden not in sql


def test_target_schema_has_required_tables_constraints_and_indexes():
    sql = read_sql("target_schema.sql").lower()
    for table in (
        "classification_run",
        "data_field_asset",
        "field_classification_result",
        "classification_evidence",
    ):
        assert f"create table {table}" in normalized(sql)
    assert "unique key uq_field_identity" in normalized(sql)
    assert "unique key uq_run_field" in normalized(sql)
    assert "on delete cascade" in normalized(sql)


def test_query_examples_cover_ten_documented_queries():
    sql = read_sql("query_examples.sql")
    assert sql.count("-- Query ") == 10
    assert "confidence < 0.75" in sql
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_sql_assets.py
```

Expected: missing SQL file failure.

- [ ] **Step 3: Create the source schema and fictional seed**

`source_schema.sql` must create `enterprise_source`, select it, and define the exact four tables
and requested columns. Every table uses a concrete comment such as `COMMENT='员工信息'`, every
column uses a concrete comment such as `COMMENT '员工内部编号'`, and
no answer field is present. Use common MySQL types: `VARCHAR`, `DATE`, `DECIMAL`, `INT`,
`DATETIME`, and primary keys.

`source_seed.sql` selects `enterprise_source` and inserts exactly five fictional rows into each
table. Use reserved demo ranges and unmistakably fictional values such as `13800000001`,
`11010019900101001X`, `6222020000000000001`, `example.test` emails, and synthetic transaction
numbers. Do not include any classification output.

- [ ] **Step 4: Create the target schema and ten queries**

`target_schema.sql` creates `compliance_result` and the four tables with every required column,
foreign key, unique constraint, status `CHECK`, and named index from the specification. Use
`JSON` only for `input_snapshot_json` and `raw_output_json`.

`query_examples.sql` must contain ten numbered queries for high-level fields by run, employee
fields, level counts, category counts, review fields, low-confidence fields, evidence by physical
field, latest successful run, high-sensitive count by table, and results by business domain.
Every query joins the actual four target tables and uses named example variables such as
`@run_id` rather than non-MySQL placeholders.

- [ ] **Step 5: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_sql_assets.py
```

Expected: all SQL structure, seed-count, constraint, index, and query tests pass.

- [ ] **Step 6: Commit Task 7**

```powershell
git add sql tests/unit/test_sql_assets.py
git commit -m "feat: add mysql demo schemas and queries"
```

### Task 8: Opt-In Real MySQL Integration Test

**Files:**
- Create: `tests/integration/test_mysql_integration.py`

- [ ] **Step 1: Write an integration test that skips safely by default**

At module load, read only process environment variables, not application settings or `.env`:

```python
SOURCE_URL = os.environ.get("MYSQL_TEST_SOURCE_URL")
TARGET_URL = os.environ.get("MYSQL_TEST_TARGET_URL")
pytestmark = pytest.mark.skipif(
    not SOURCE_URL or not TARGET_URL,
    reason="MYSQL_TEST_SOURCE_URL and MYSQL_TEST_TARGET_URL are not configured",
)
```

Before executing schema SQL, parse both URLs and assert their database names contain `test`.
Load the tracked SQL text, replace only the exact identifiers `enterprise_source` and
`compliance_result` with the parsed dedicated test database names, and then execute it. The test
must execute the source schema and seed, execute the target schema, instantiate both real
repositories, scan the four tables, persist one deterministic fake classification record, then
verify `get_run`, `query_results`, and `get_result_evidence` return the inserted values.

- [ ] **Step 2: Run without database URLs and verify explicit skip**

```powershell
Remove-Item Env:MYSQL_TEST_SOURCE_URL -ErrorAction SilentlyContinue
Remove-Item Env:MYSQL_TEST_TARGET_URL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pytest -q tests\integration\test_mysql_integration.py -rs
```

Expected: one skipped test with the configured reason; exit code 0.

- [ ] **Step 3: Run the ordinary suite and verify no external dependency**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all ordinary tests pass and the MySQL test is skipped.

- [ ] **Step 4: Commit Task 8**

```powershell
git add tests/integration/test_mysql_integration.py
git commit -m "test: add opt-in mysql integration coverage"
```

### Task 9: README, Contract Consistency, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md` only if test commands need clarification
- Review: all created and modified files

- [ ] **Step 1: Update README to the actual phase-two system**

Document these exact responsibilities and commands:

```text
A enterprise_source MySQL: enterprise business tables and sample values
Chroma: regulatory knowledge chunks
B compliance_result MySQL: run, field asset, result, and evidence records
```

Include venv installation, both database URLs, SQL initialization order, knowledge rebuild,
FastAPI startup, `/api/pipeline/run` PowerShell example, result endpoints, location of query
examples, ordinary test command, and opt-in MySQL test environment variables. Keep the existing
manual `/api/classify` example and remove the old statement that database persistence is outside
the current phase.

- [ ] **Step 2: Run first full verification round**

```powershell
.\.venv\Scripts\python.exe -m compileall app scripts
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: compileall and Ruff exit 0; all ordinary tests pass and MySQL integration reports skip.

- [ ] **Step 3: Perform schema-to-code consistency audit**

Check exact names across:

```text
FieldProfile ↔ SourceMySQLRepository metadata mapping
FieldClassificationRecord ↔ TargetMySQLRepository insert parameters
TargetMySQLRepository ↔ sql/target_schema.sql
API response models ↔ repository return models
sql/query_examples.sql ↔ sql/target_schema.sql
.env.example ↔ Settings
```

Run focused fake-chain tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_database_pipeline.py tests\unit\test_target_record_mapping.py tests\integration\test_pipeline_api.py
```

Expected: all focused tests pass; successive run IDs differ, field IDs match, counters and
evidence order match assertions.

- [ ] **Step 4: Scan forbidden legacy names and dependencies**

Use PowerShell `Select-String` over tracked Python, Markdown, TOML, SQL, and environment-example
files for:

```text
mysql_evaluator|result_store|batch_evaluator|MYSQL_HOST|MYSQL_DATABASE|api_key.env|Week3|Gradio|Conda|backend\.|save_results\(|classification_results
```

Expected: no active-code or documentation matches. Confirm requirements contain SQLAlchemy and
PyMySQL and contain none of the removed Gradio/Excel/MySQL-evaluator dependencies.

- [ ] **Step 5: Run final verification and inspect the complete diff**

```powershell
.\.venv\Scripts\python.exe -m compileall app scripts
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m pip check
git diff --check
git status --short --branch
git diff --stat
git diff
```

Expected: every command exits 0; ordinary tests pass, MySQL integration is explicitly skipped
when URLs are absent, dependency check is clean, and Git shows only phase-two source, SQL,
documentation, configuration, and test changes.

- [ ] **Step 6: Commit final documentation or consistency fixes**

```powershell
git add README.md CONTRIBUTING.md app tests sql .env.example requirements.txt requirements-dev.txt
git commit -m "docs: document mysql classification pipeline"
```

Do not push. Report real MySQL, model, and external LLM checks separately and never claim an
unconfigured integration test passed.
