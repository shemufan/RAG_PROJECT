# A/B Database Classification Pipeline Design

## Goal

Build the phase-two demo workflow that reads physical field metadata and masked samples from
the `enterprise_source` MySQL database, classifies each field through the existing RAG service,
and persists queryable assets, runs, results, and evidence in the independent
`compliance_result` MySQL database.

## Scope

This phase adds synchronous MySQL scanning, orchestration, persistence, SQL initialization,
query APIs, and tests. It preserves the existing health endpoint, single-field classification
endpoint, Chroma knowledge retrieval, structured LLM output, and knowledge-base rebuild script.

It does not add background workers, queues, agents, frontends, reports, migrations, container
orchestration, or generic database plugin abstractions.

## Architecture

The implementation uses SQLAlchemy 2.x Core and PyMySQL. It does not introduce ORM entity
relationships. Each component has one responsibility:

- `SourceMySQLRepository` reads source metadata and samples and maps them to `FieldProfile`.
- `FieldClassificationService` remains the only RAG and LLM classification component.
- `DatabaseClassificationPipeline` coordinates one scan run without embedding SQL.
- `TargetMySQLRepository` persists and queries compliance records without calling RAG or LLM.
- `pipeline` API routes validate HTTP input and delegate to the pipeline or target repository.

Both database connections come from independent `SOURCE_DATABASE_URL` and
`TARGET_DATABASE_URL` settings. They are constructed lazily when a pipeline endpoint is used,
so `/api/health` and `/api/classify` do not require MySQL.

## Data Flow

1. The API validates a `PipelineRequest` and resolves a pipeline dependency.
2. The pipeline creates a UUID4 run identifier and persists a `RUNNING` run.
3. The source repository reads `information_schema.COLUMNS` and
   `information_schema.TABLES` for the selected tables.
4. Each trusted metadata row is mapped to a strict `FieldProfile`; sample queries use only
   identifiers returned by the metadata query and SQLAlchemy identifier quoting.
5. Samples are non-null, capped at five, truncated to fifty characters after masking, and never
   copied into logs.
6. The existing classification service performs Chroma retrieval and structured LLM inference.
7. A successful L1-L4 result is converted to `ClassificationOutput`, combined with the field,
   evidence, model metadata, UUID5 field ID, and run ID as `FieldClassificationRecord`.
8. The target repository upserts the field asset, then saves the result and ordered evidence in
   one transaction.
9. The pipeline updates final run counters and status and returns `PipelineSummary`.

The UUID5 field identifier uses the namespace URL UUID and the exact lower-level identity string
`source_system:database_name:table_name:field_name`. The same physical field therefore keeps its
identifier across runs, while each run receives a new UUID4.

## Schemas

`FieldProfile` gains schema version `1.0`, required source/database/table/field/data-type values,
table and column metadata, nullability, key information, masked samples, and business domain.
It contains no credentials or classification answer.

`ClassificationOutput` remains limited to category, subcategory, L1-L4 level, confidence,
reason, and review flag. `Evidence` remains a structured regulatory citation.

`FieldClassificationRecord` combines one validated input and successful output with evidence,
stable identity, run identity, decision path, model name, knowledge-base version, and timestamp.
`PipelineRequest`, `PipelineSummary`, run/result query responses, and evidence responses are
separate HTTP-facing Pydantic models.

## Source Database

`enterprise_source` contains four ordinary business tables: `employee`, `customer_account`,
`customer_order`, and `product`. Tables and columns have comments, snake-case names, and only
business data. No category, level, or sensitivity answer is stored in the source.

The seed file inserts five wholly fictional rows per table. Identity numbers, phone numbers,
bank cards, addresses, accounts, and transaction identifiers are synthetic. All samples still
pass through the same masking path used for non-demo data.

The source repository uses a stable table-to-domain mapping: employee to `hr`, customer account
to `customer`, customer order to `commerce`, product to `product`, and all other tables to
`general`. A failed sample query produces an empty sample list for that field and does not abort
the metadata scan. Metadata scan failures are raised to the pipeline.

## Masking

- Chinese identity numbers retain the first four and last four characters.
- Mobile numbers retain the first three and last four characters.
- Bank cards retain the first four and last four characters.
- Email local parts retain only their first character while the domain is preserved.
- Other values are converted to strings and capped at fifty characters.

Field-name hints select identity, phone, bank-card, and email masking. Returned samples are
deduplicated without changing their first-seen order.

## Target Database

`compliance_result` contains four relational tables:

- `classification_run`: lifecycle, counters, model and knowledge-base metadata.
- `data_field_asset`: stable physical field identity and current metadata timestamps.
- `field_classification_result`: one successful classification per run and field, including
  queryable conclusion columns plus JSON audit snapshots.
- `classification_evidence`: ordered regulatory evidence linked to a result with cascade delete.

Frequently filtered properties remain ordinary indexed columns. JSON is used only for the exact
validated input snapshot and structured output audit snapshot.

`upsert_field_asset` refreshes metadata and `last_seen_at` while preserving `first_seen_at`.
`save_classification_record` writes the result and all evidence in one transaction. Repository
write errors are raised with operation context and are never converted to zero or success.

## Failure Semantics

The existing single-field service may return `UNKNOWN` after an internal dependency failure.
The pipeline treats `UNKNOWN` as a failed field: it upserts the field asset, increments
`failed_fields`, and does not insert a classification result or evidence. This preserves the
strict L1-L4 `ClassificationOutput` contract.

With `continue_on_error=true`, field failures are logged and subsequent fields continue. With
`continue_on_error=false`, processing stops after the first failed field. Already committed
field transactions remain valid, and the run is finalized from the accumulated counters.

Run status rules are:

- `SUCCESS`: every scanned field produced and persisted a successful result.
- `PARTIAL_FAILED`: at least one field succeeded and at least one failed.
- `FAILED`: scanning failed, no fields were scanned, or every processed field failed.

`need_review=true` increments `review_fields` but is not an execution failure. A scan failure is
recorded in the run error message, the run is finalized as `FAILED`, and the API receives a
failed `PipelineSummary`. A failure to create or finalize the run itself is raised because the
target database can no longer provide a truthful task record.

## API

- `POST /api/pipeline/run` runs the synchronous pipeline in FastAPI's worker thread.
- `GET /api/runs/{run_id}` returns one run or HTTP 404.
- `GET /api/results` supports bounded pagination and typed optional filters.
- `GET /api/results/{result_id}/evidence` returns evidence in `rank_no` order.

API code contains no SQL and never calls Chroma or the LLM directly. Database URLs are never
accepted from requests.

## SQL Deliverables

The `sql` directory contains source schema creation, fictional source seed data, target schema
creation with constraints and indexes, and ten executable query examples matching actual names.
The documented execution order is source schema, source seed, target schema, knowledge rebuild,
then API startup.

## Testing

Unit tests use fake engines, repositories, and classifier objects. They cover metadata mapping,
nullability, domains, sample limits, masking, stable IDs, record construction, target parameter
mapping, all pipeline status paths, review counts, and evidence ordering.

API tests override dependencies and verify the four pipeline/query routes plus the existing
classification route. The MySQL integration test is skipped unless dedicated test database URLs
are configured; when enabled, it creates the documented schemas, scans, persists, and queries.

Verification runs compileall, Ruff, the complete pytest suite, schema-to-SQL consistency checks,
legacy-name scans, and a final Git diff review. Lack of a real MySQL server, LLM key, or model is
reported explicitly rather than represented as a successful integration run.
