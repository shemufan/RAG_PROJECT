# Experiment E: Semantic Bridge RAG Design

## 1. Goal

Implement Experiment E on top of commit `2e99381` from
`experiment/query-ablation` to test whether the limited gain of regulation-only
RAG is caused by a semantic gap between raw field data and regulatory text.

The experiment must preserve the benchmark, labels, regulation knowledge base,
regulation chunks, embedding model, LLM model, temperature, regulation Top-K,
classification output schema, and evaluation logic used by Experiment B. Its only
material change is the Semantic Bridge stage:

```text
Field
-> objective Value Profiling
-> Semantic KB retrieval
-> Top-1 semantic_type
-> regulation query reconstruction
-> existing regulation RAG
-> existing LLM classification
-> existing evaluation
```

Experiment E is diagnostic. It must not contain benchmark-specific cards, hidden
ground-truth labels, semantic classification rules, or an additional LLM call.

## 2. Baseline and experiment mapping

The existing CSV runner exposes the relevant baselines as parameter combinations:

- B: `--query-strategy clean --use-rag`
- C: `--query-strategy profile --use-rag`
- D: `--query-strategy clean --no-use-rag`

Experiment E adds `--experiment E`. Existing options and their defaults remain
valid so that A/B/C/D behavior is unchanged. E uses the same regulation store and
`regulation_top_k=3` as B. The Semantic KB has its own default
`semantic_top_k=3`.

## 3. Architecture

### 3.1 Components

The implementation adds focused Semantic Bridge components instead of making the
regulation repository interpret semantic cards:

- `SemanticCard`: validates one semantic type and its descriptive fields.
- Semantic KB loader: reads tracked JSON cards, rejects duplicate semantic types,
  and converts every card to one natural-language LangChain `Document`.
- `SemanticVectorStore`: owns a Chroma collection and persistence path that are
  distinct from the regulation collection and path. It exposes raw retrieval
  scores together with reconstructed cards.
- objective profile builder: calls only `ValueProfiler.basic_statistics()`.
  It never calls `ValueProfiler.profile()` and therefore never consumes
  `candidate_types` or detector conclusions.
- semantic query builder: serializes `field_name`, optional `field_cn`, sample
  values, and objective profile features.
- regulation bridge query builder: serializes `field_name`, optional `field_cn`,
  selected `semantic_type`, `semantic_category`, and `regulation_keywords`.
  Raw samples and profile features are intentionally excluded.
- `SemanticBridgeClassificationService`: orchestrates the two retrievals and the
  existing LLM call, and returns both the stable classification result and an E
  trace used by exporters.
- experiment exporter: writes per-case results, the metric summary, and a compact
  semantic retrieval audit table.

The existing `VectorStore` remains the regulation store. It gains only the
minimum raw-result interface needed to record unmodified regulation retrieval
scores; its current `search()` contract remains unchanged for A/B/C/D.

### 3.2 Data flow

For every `FieldProfile`:

1. Validate the input through the existing schema.
2. Compute generic facts through `ValueProfiler.basic_statistics(samples)`.
3. Build the Semantic Query from names, samples, and generic facts.
4. Retrieve three Semantic Cards from the independent Semantic collection,
   retaining each raw score.
5. Select the first returned card without thresholding or reranking.
6. Build the regulation Query from the selected card's domain semantics.
7. Retrieve three unchanged regulation chunks from the existing regulation
   collection, retaining raw score, source, chunk id, and content.
8. Build the existing structured classification prompt with two additive context
   sections: objective value profile and selected Semantic Card.
9. Invoke the existing `LLMService` once and validate the same
   `ClassificationOutput` schema.
10. Evaluate predictions through the existing benchmark evaluator.

No semantic-stage LLM call or rule fallback is allowed. Empty semantic or
regulation retrieval is a visible case failure rather than a silent change to the
experiment path.

## 4. Profiling responsibility

Experiment E treats the existing `ValueProfiler` as two conceptual layers. Only
its public `basic_statistics()` layer is allowed. Valid E profile features include:

- approximate or typical length;
- length consistency and format consistency;
- numeric, alphabetic, Chinese, or hexadecimal character ratios;
- presence of masking characters or generic separators/symbols;
- other facts that describe visible structure without naming a domain type.

Experiment E must not call or reproduce the existing mobile, email, IP, identity
card, bank card, device, coordinate, or date detectors. It must not use
`candidate_types`, field-name hint rules, regex-based semantic predictions, or an
`if length == N` semantic mapping.

The design therefore preserves the required responsibility boundary:

```text
Profiling: what the field looks like
Semantic KB retrieval: what the field may mean
```

## 5. Semantic Knowledge Base

### 5.1 Tracked source format

The first version is a tracked UTF-8 JSON array containing roughly 30-40 general
cards. Each card has this schema:

```json
{
  "semantic_type": "手机号码",
  "aliases": ["手机号", "联系电话", "phone", "mobile"],
  "common_field_names": ["phone", "mobile", "contact_phone"],
  "value_features": ["通常由数字组成", "中国大陆手机号通常为11位"],
  "description": "用于联系自然人的电话号码",
  "semantic_category": ["联系方式", "个人信息"],
  "regulation_keywords": ["手机号码", "电话号码", "联系方式", "个人信息"]
}
```

Required strings must be non-empty and list fields must contain unique,
non-empty strings. `semantic_type` is unique across the file.

Initial coverage includes names, mobile and fixed-line phones, identity document
numbers, email, address, bank card, financial account, password, IP, MAC, device
ID, location, face, fingerprint, salary, order number, contract number, enterprise
name, unified social credit code, medical information, date, time, user ID,
customer number, and other common generic business identifiers.

Cards are authored from general domain knowledge only. They do not contain
benchmark values or benchmark labels. Any examples are synthetic and masked.

### 5.2 Embedding text

Each card becomes a deterministic natural-language document:

```text
语义类型：手机号码
别名：手机号、联系电话、phone、mobile
常见字段名：phone、mobile、contact_phone
数据特征：通常由数字组成；中国大陆手机号通常为11位
业务含义：用于联系自然人的电话号码
语义类别：联系方式、个人信息
法规关键词：手机号码、电话号码、联系方式、个人信息
```

The document metadata contains the full card fields plus a stable semantic card
identifier. It uses the existing `EmbeddingService`, so E introduces no model
change.

### 5.3 Storage isolation and rebuild

Semantic vectors use settings distinct from the regulation store, for example:

- persistence directory: `.runtime/semantic_chroma`
- collection: `semantic_field_types`

The existing regulation settings remain untouched. A dedicated rebuild command
validates all cards, resets only the Semantic collection, adds the card documents,
and reports card count. It never resets or writes the regulation collection.

## 6. Query contracts

### 6.1 Semantic Query

The deterministic Semantic Query contains:

```text
字段名称：contact_value
中文名称：联系值
样本值：13812345678、15987654321
字段客观特征：字符串长度约11位；长度一致；多个样例格式一致；主要由数字组成；数字字符占比100%
```

Missing `field_cn` is omitted. Empty sample input is represented by the objective
profile's `无有效样例` fact.

### 6.2 Regulation Query

The selected Top-1 card produces:

```text
字段：contact_value
中文名称：联系值
字段语义类型：手机号码
语义类别：联系方式、个人信息
法规检索关键词：手机号码、电话号码、联系方式、个人信息
```

Samples and objective profile features do not enter this query. This makes the
Semantic Card, rather than raw digits, the bridge to regulatory language.

## 7. Classification prompt compatibility

The system prompt, model construction, temperature, output schema, and final
decision rules remain unchanged. The E human message contains:

1. the original validated field profile, including samples;
2. objective value profile features;
3. the selected Semantic Card and its raw score;
4. the existing regulation evidence list.

The two new sections are additive and clearly marked as untrusted data. No new
classification instructions, label definitions, examples, or reasoning procedure
are introduced.

## 8. Scores and diagnostic trace

Semantic and regulation stores expose the exact float returned by the underlying
similarity API as `raw_score`. It is never clipped to `[0, 1]`, thresholded, or
replaced with a post-threshold value. Existing normalized `Evidence.score` may
remain for compatibility, but E diagnostics are populated from raw retrieval
rows.

Each E case records:

```json
{
  "field_name": "contact_value",
  "sample_values": ["13812345678", "15987654321"],
  "profiling": {"features": ["字符串长度约11位", "长度一致"]},
  "semantic_query": "...",
  "semantic_retrieval": [
    {"semantic_type": "手机号码", "raw_score": 0.82}
  ],
  "selected_semantic_type": "手机号码",
  "regulation_query": "...",
  "regulation_retrieval": [
    {"chunk": "...", "raw_score": 0.71, "source": "..."}
  ],
  "prediction": true,
  "ground_truth": true,
  "correct": true
}
```

The trace also retains aliases, semantic category, regulation keywords,
description, chunk id, and classification details needed for later layer-by-layer
analysis.

## 9. Experiment output

Every new E run creates:

```text
outputs/experiment_E/<run_id>/experiment_E_results.csv
outputs/experiment_E/<run_id>/experiment_E_summary.json
outputs/experiment_E/<run_id>/semantic_retrieval_debug.csv
```

`experiment_E_results.csv` is one row per benchmark case and includes serialized
profile, queries, retrieval arrays, prediction, ground truth, outcome, correctness,
and any failure type.

`experiment_E_summary.json` contains the run identity, immutable E parameters,
model/knowledge versions, TP/FP/TN/FN, Precision, Recall, F1, Accuracy, Coverage,
Effective Recall, case counts, and output paths.

`semantic_retrieval_debug.csv` is optimized for human review with field, samples,
profile, Top-1/2/3 semantic types and raw scores, selected type, and Top1 minus
Top2 score gap. When fewer than three cards are returned, missing cells remain
empty. No semantic accuracy metric is fabricated because the dataset has no
semantic-type ground truth.

Files are written as UTF-8 with BOM for convenient Chinese text inspection in
spreadsheet software. A run-specific directory prevents accidental overwrite.

## 10. CLI and configuration

The normal E execution is:

```powershell
G:\ANACONDA\develop\envs\rag_env\python.exe -m scripts.run_csv_pipeline `
  --experiment E `
  --input "RAG_mini_benchmark_150.csv" `
  --input-mode catalog `
  --label-column expected_personal
```

The runner accepts `--semantic-top-k 3` and an optional output root while keeping
the E defaults above. Invalid combinations fail early: Experiment E always uses
RAG, objective profiling, the Semantic Bridge, no semantic LLM, and the B
regulation Top-K. Resume must use the same experiment and parameters; otherwise it
is rejected rather than mixing incomparable results.

A separate command rebuilds the Semantic KB before the first E run. The E runner
checks that both vector stores are non-empty and gives the exact rebuild command
when the Semantic store is missing.

## 11. Failure behavior

- Invalid or duplicate Semantic Cards stop rebuild before resetting the existing
  Semantic collection.
- An empty Semantic collection stops E startup.
- Empty Semantic retrieval marks the case failed and records the semantic query.
- Empty regulation retrieval marks the case failed and preserves all preceding
  semantic diagnostics.
- LLM or output-validation failure follows the existing failed-case behavior and
  preserves both retrieval traces.
- Export failure surfaces as a run-level error; it does not silently claim a
  complete experiment without the requested artifacts.

Logs and exported failures contain exception class names and stage names, but no
API keys, database credentials, or full environment configuration.

## 12. Testing strategy

Implementation follows red-green-refactor cycles. Automated tests use injected
fake embeddings, vector stores, repositories, and structured models, so they do
not call a paid LLM or require a production database.

Tests verify:

- all Semantic Cards validate, cover the required common types, and contain no
  benchmark-derived data;
- deterministic card text and metadata construction;
- Semantic rebuild touches only the Semantic collection;
- E profiling uses generic facts and never exposes candidate semantic types;
- Semantic Query uses names, samples, and objective facts;
- Top-K retrieval preserves raw scores and Top-1 selection;
- regulation Query contains selected semantic meaning and excludes samples;
- the existing regulation RAG path runs with B's Top-K;
- the E prompt includes field, samples, profile, Semantic Card, and regulation
  evidence while retaining the existing system prompt and output schema;
- the LLM is invoked exactly once per successful E case;
- all three output artifacts contain the required diagnostic and metric fields;
- metrics are generated through the existing evaluator;
- existing A/B/C/D CLI combinations and service behavior remain unchanged;
- an end-to-end injected-dependency smoke test covers Field -> Profile -> Semantic
  RAG -> regulation RAG -> LLM -> Prediction -> Metrics -> Artifacts.

The baseline in the requested `rag_env` is 236 passing and 2 skipped tests when
pytest is run with `-o "filterwarnings="`; the override is needed because the
environment's Starlette version no longer exports the warning class named by the
current project configuration. This pre-existing warning-filter incompatibility
is not part of Experiment E's production behavior.

## 13. Acceptance criteria

Experiment E is complete when:

1. branch `experiment/E-part` is based on `experiment/query-ablation` commit
   `2e99381`;
2. the regulation and Semantic vector stores are demonstrably independent;
3. the Semantic KB contains approximately 20-50 general cards without benchmark
   values or labels;
4. objective profiling cannot emit a semantic type in the E path;
5. Semantic Top-3 retrieval and raw scores are recorded and Top-1 is selected;
6. the selected semantic meaning enters the regulation Query while raw samples do
   not;
7. the existing regulation Top-K, LLM, prompt contract, and evaluator are retained
   except for additive E context;
8. the requested three run-isolated artifacts are generated;
9. A/B/C/D regression tests and all E tests pass in `rag_env`;
10. documented rebuild and E commands are executable with the user's local model,
    database, and API configuration.

