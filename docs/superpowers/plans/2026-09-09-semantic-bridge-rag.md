# Semantic Bridge RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Experiment E, which maps objective field profiles through an independent Semantic Knowledge Base before retrieving unchanged regulation evidence and invoking the existing classifier.

**Architecture:** Add validated Semantic Card models, deterministic query builders, and a dedicated Chroma adapter. A Semantic Bridge classification service composes those pieces with the existing regulation store and LLM, while an optional pipeline observer exports run-isolated diagnostics without changing A/B/C/D behavior.

**Tech Stack:** Python 3.10 (`rag_env`), Pydantic, LangChain Documents, Chroma, argparse, pytest, CSV/JSON.

---

## File map

- Create `data/semantic_knowledge/semantic_cards.json`: general semantic cards.
- Create `app/schemas/semantic.py`: Semantic Card, retrieval result, and trace models.
- Create `app/services/semantic_knowledge_service.py`: card loading, document conversion, and rebuild orchestration.
- Create `app/repositories/semantic_vector_store.py`: independent Semantic Chroma collection.
- Create `app/rag/semantic_bridge.py`: objective Semantic Query and regulation Query builders.
- Create `app/services/semantic_bridge_service.py`: two-stage retrieval and one-call LLM orchestration.
- Create `app/services/experiment_e_reporter.py`: three requested run artifacts.
- Create `scripts/rebuild_semantic_knowledge_base.py`: Semantic KB rebuild CLI.
- Modify `app/core/config.py`: independent Semantic KB settings.
- Modify `.env.example` and `README.md`: document environment keys and commands.
- Modify `app/repositories/vector_store.py`: add raw regulation retrieval while retaining `search()`.
- Modify `app/rag/prompt.py`: optional additive E context.
- Modify `app/schemas/classification.py`: attach an excluded internal trace to results.
- Modify `app/services/csv_pipeline.py`: optional experiment observer hooks.
- Modify `scripts/run_csv_pipeline.py`: A-E presets and E dependency wiring.
- Add focused unit and integration tests under `tests/unit` and `tests/integration`.

### Task 1: Semantic Card schema, dataset, and document loader

**Files:**
- Create: `app/schemas/semantic.py`
- Create: `app/services/semantic_knowledge_service.py`
- Create: `data/semantic_knowledge/semantic_cards.json`
- Test: `tests/unit/test_semantic_knowledge.py`

- [ ] **Step 1: Write failing schema and loader tests**

Cover valid cards, duplicate semantic types, empty strings, deterministic natural-language text, metadata reconstruction, a card count between 20 and 50, and required common types. Assert that a repository-wide scan of the Semantic JSON contains no benchmark label keys.

```python
def test_semantic_cards_load_and_cover_common_types():
    cards = load_semantic_cards(PROJECT_ROOT / "data/semantic_knowledge/semantic_cards.json")
    assert 20 <= len(cards) <= 50
    names = {card.semantic_type for card in cards}
    assert {"手机号码", "身份证号码", "邮箱", "IP地址", "用户ID"} <= names

def test_card_document_is_deterministic():
    document = semantic_card_to_document(phone_card)
    assert document.page_content.startswith("语义类型：手机号码\n别名：")
    assert document.metadata["semantic_type"] == "手机号码"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
G:\ANACONDA\develop\envs\rag_env\python.exe -m pytest tests/unit/test_semantic_knowledge.py -q -o "filterwarnings="
```

Expected: collection fails because semantic modules and data do not exist.

- [ ] **Step 3: Implement the models, loader, and 30-40 general cards**

Use required non-empty strings and de-duplicated lists. `semantic_card_to_document()` must include every card field in deterministic label order. Metadata contains JSON strings for list fields so Chroma receives scalar metadata values.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all semantic knowledge tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/schemas/semantic.py app/services/semantic_knowledge_service.py data/semantic_knowledge/semantic_cards.json tests/unit/test_semantic_knowledge.py
git commit -m "feat: add semantic field knowledge cards"
```

### Task 2: Independent Semantic vector store and configuration

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Create: `app/repositories/semantic_vector_store.py`
- Create: `scripts/rebuild_semantic_knowledge_base.py`
- Test: `tests/unit/test_semantic_vector_store.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing configuration and repository tests**

Assert these defaults resolve below the project root and differ from regulation settings:

```python
assert settings.semantic_chroma_db_dir == root / ".runtime/semantic_chroma"
assert settings.semantic_chroma_collection == "semantic_field_types"
assert settings.semantic_knowledge_file == root / "data/semantic_knowledge/semantic_cards.json"
assert settings.semantic_chroma_db_dir != settings.chroma_db_dir
assert settings.semantic_chroma_collection != settings.chroma_collection
```

Use an injected fake LangChain store to assert `search(query, k=3)` returns full Semantic Cards and exact unbounded `raw_score` values, including a score such as `1.2`.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: missing settings and repository imports.

- [ ] **Step 3: Implement independent storage and rebuild CLI**

`SemanticVectorStore` constructs Chroma with only Semantic settings. Its `reset()` recreates only the Semantic collection. The rebuild command loads and validates all cards before calling reset.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: config, store, and rebuild tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/core/config.py .env.example app/repositories/semantic_vector_store.py scripts/rebuild_semantic_knowledge_base.py tests/unit/test_semantic_vector_store.py tests/unit/test_config.py
git commit -m "feat: add independent semantic vector store"
```

### Task 3: Objective profiling and two deterministic queries

**Files:**
- Create: `app/rag/semantic_bridge.py`
- Test: `tests/unit/test_semantic_bridge.py`

- [ ] **Step 1: Write failing query tests**

Inject a profiler whose `profile()` raises and whose `basic_statistics()` returns
generic features. Assert only `basic_statistics()` is called.

```python
class ObjectiveOnlyProfiler:
    def profile(self, *_args):
        raise AssertionError("semantic candidates are forbidden")

    def basic_statistics(self, values):
        assert values == ["13812345678", "15987654321"]
        return ["字符串长度约11位", "长度一致", "数字字符占比100%"]
```

Assert the Semantic Query contains `field_name`, `field_cn`, samples, and generic
features. Assert the regulation Query contains the selected card fields but none
of the samples or generic feature strings.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: missing query builder module.

- [ ] **Step 3: Implement minimal builders**

Provide `build_objective_profile(field, profiler)`,
`build_semantic_query(field, features)`, and
`build_regulation_bridge_query(field, card)`. Do not add pattern detectors,
candidate types, thresholds, or semantic fallbacks.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit**

```powershell
git add app/rag/semantic_bridge.py tests/unit/test_semantic_bridge.py
git commit -m "feat: build semantic bridge queries from objective facts"
```

### Task 4: Raw regulation retrieval and additive E prompt

**Files:**
- Modify: `app/repositories/vector_store.py`
- Modify: `app/rag/prompt.py`
- Modify: `app/schemas/classification.py`
- Test: `tests/unit/test_services.py`
- Test: `tests/unit/test_rag.py`

- [ ] **Step 1: Write failing raw-score and prompt tests**

Assert `search_raw()` retains `1.2` as `raw_score` while existing `search()` still
maps it to compatible `Evidence.score == 1.0`. Assert E prompt context includes
objective features and selected Semantic Card but the system prompt remains byte
for byte equal to `CLASSIFICATION_SYSTEM_PROMPT`.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Add backward-compatible raw rows and optional prompt parameters**

Implement one internal raw regulation retrieval model. Keep `VectorStore.search()`
signature and output unchanged. Extend `build_classification_prompt()` with
keyword-only optional `value_profile` and `semantic_knowledge` arguments; omitted
arguments reproduce existing prompt output exactly.

- [ ] **Step 4: Run focused and existing service tests**

Expected: all old and new RAG/service tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/repositories/vector_store.py app/rag/prompt.py app/schemas/classification.py tests/unit/test_services.py tests/unit/test_rag.py
git commit -m "feat: preserve raw retrieval scores for experiment E"
```

### Task 5: Two-stage Semantic Bridge classification service

**Files:**
- Create: `app/services/semantic_bridge_service.py`
- Test: `tests/unit/test_semantic_bridge_service.py`

- [ ] **Step 1: Write failing orchestration tests**

Use capturing stores and an injected LLM. Verify exact order and contracts:

```text
basic_statistics -> semantic search(k=3) -> select result[0]
-> regulation search_raw(k=3) -> build E prompt -> one LLM call
```

Assert the trace retains Top-3 cards, raw scores, score gap, queries, regulation
chunks, and selected Top-1 card. Assert no LLM call occurs when either retrieval
is empty and the returned classification is `UNKNOWN` with a trace identifying
the failed stage.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement the E service**

Mirror existing `FieldClassificationService` output mapping and failure behavior.
Set `decision_path="semantic_rag_llm"` on success and
`"semantic_rag_llm_error"` on failure. Attach the trace through a Pydantic field
excluded from normal response serialization.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit**

```powershell
git add app/services/semantic_bridge_service.py tests/unit/test_semantic_bridge_service.py app/schemas/classification.py app/schemas/semantic.py
git commit -m "feat: orchestrate semantic bridge classification"
```

### Task 6: Run observer and requested artifacts

**Files:**
- Create: `app/services/experiment_e_reporter.py`
- Modify: `app/services/csv_pipeline.py`
- Test: `tests/unit/test_experiment_e_reporter.py`
- Test: `tests/unit/test_csv_pipeline.py`

- [ ] **Step 1: Write failing reporter and pipeline-hook tests**

Use a fixed UUID and temporary output root. Record one successful and one failed
case, finalize with existing metrics, then assert these exact paths exist:

```text
outputs/experiment_E/<run_id>/experiment_E_results.csv
outputs/experiment_E/<run_id>/experiment_E_summary.json
outputs/experiment_E/<run_id>/semantic_retrieval_debug.csv
```

Assert UTF-8 BOM, required JSON/CSV columns, raw scores, Top1-Top2 gap, ground
truth, correctness, TP/FP/TN/FN, Precision, Recall, F1, and Accuracy. Verify a
pipeline with no observer follows its old call behavior.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement an optional observer and atomic exporters**

Add no-op-by-default hooks for run start, per-case completion, and final summary.
The E reporter writes through temporary sibling files and replaces final files
only after complete serialization. Use `utf-8-sig` for CSV and JSON artifacts.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit**

```powershell
git add app/services/experiment_e_reporter.py app/services/csv_pipeline.py tests/unit/test_experiment_e_reporter.py tests/unit/test_csv_pipeline.py
git commit -m "feat: export experiment E diagnostics"
```

### Task 7: A-E CLI presets and E dependency wiring

**Files:**
- Modify: `scripts/run_csv_pipeline.py`
- Test: `tests/integration/test_csv_pipeline_cli.py`
- Test: `tests/integration/test_experiment_e_smoke.py`

- [ ] **Step 1: Write failing CLI tests**

Assert `--experiment` accepts A, B, C, D, and E with mappings:

```python
EXPERIMENT_PRESETS = {
    "A": ("legacy", "c", "rule", True),
    "B": ("clean", "c", "rule", True),
    "C": ("profile", "c", "rule", True),
    "D": ("clean", "c", "rule", False),
}
```

Assert E creates both stores with one embedding service, checks both counts,
constructs `SemanticBridgeClassificationService` with semantic Top-K 3 and
regulation Top-K 3, and injects `ExperimentEReporter`. Existing calls without
`--experiment` must retain all current defaults and validation.

The smoke test uses fakes and verifies Field -> profile -> Semantic Top-1 ->
regulation query -> regulation evidence -> one LLM output -> metrics -> three
files.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement presets and E wiring**

Add `--experiment {A,B,C,D,E}`, `--semantic-top-k` default 3, and
`--output-root` default `outputs`. E ignores no settings silently: incompatible
explicit flags are rejected. Keep direct legacy flags operational when
`--experiment` is absent.

- [ ] **Step 4: Run CLI and smoke tests and verify GREEN**

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py tests/integration/test_experiment_e_smoke.py
git commit -m "feat: expose semantic bridge experiment E"
```

### Task 8: Documentation and full regression verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Test: all tests

- [ ] **Step 1: Add documentation assertions if command text is tested**

Verify documentation contains the Semantic rebuild command, E execution command,
output paths, comparison commands for B/D/E, fairness constraints, and the
objective-profiling prohibition.

- [ ] **Step 2: Update README and environment example**

Document commands with the requested interpreter:

```powershell
G:\ANACONDA\develop\envs\rag_env\python.exe -m scripts.rebuild_semantic_knowledge_base
G:\ANACONDA\develop\envs\rag_env\python.exe -m scripts.run_csv_pipeline --experiment E --input "RAG_mini_benchmark_150.csv" --input-mode catalog --label-column expected_personal
```

- [ ] **Step 3: Run static checks**

```powershell
git diff --check
G:\ANACONDA\develop\envs\rag_env\python.exe -m compileall -q app scripts tests
```

Expected: both exit zero.

- [ ] **Step 4: Run the complete suite**

```powershell
G:\ANACONDA\develop\envs\rag_env\python.exe -m pytest -q -o "filterwarnings="
```

Expected: all tests pass, with only explicitly skipped external integration tests.

- [ ] **Step 5: Verify requirements against the design**

Inspect the diff and confirm every acceptance criterion in
`docs/superpowers/specs/2026-09-09-semantic-bridge-rag-design.md` has code or test
evidence. Confirm no benchmark file or value has entered the Semantic KB and no
new LLM call or semantic prediction rule exists.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md .env.example
git commit -m "docs: document semantic bridge experiment"
```
