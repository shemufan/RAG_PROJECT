# No-RAG Evidence Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment D so the CSV benchmark can skip `VectorStore.search()` while preserving B's clean strategy, complete FieldProfile, classification Prompt, LLM, output schema, and evaluation.

**Architecture:** Add a default-on `use_rag` switch at the classification orchestration boundary. The RAG path remains byte-for-byte equivalent in behavior; the no-RAG path supplies `evidence=[]` to the existing Prompt builder and then rejoins the unchanged LLM/result path. Expose the switch only through the CSV experiment CLI as `--use-rag/--no-use-rag`.

**Tech Stack:** Python 3.11+, argparse, Pydantic, LangChain message models, pytest, Ruff

---

## File map

- Modify `app/services/classification_service.py`: own the optional retrieval step and preserve the shared classification path.
- Modify `scripts/run_csv_pipeline.py`: expose and validate the experiment-D CLI switch, then inject it into the service.
- Modify `tests/unit/test_services.py`: prove D performs no query construction/search and uses the unchanged Prompt with empty Evidence.
- Modify `tests/integration/test_csv_pipeline_cli.py`: prove CLI defaults, validation, and dependency propagation.
- Do not modify `app/rag/prompt.py`, retrieval builders, VectorStore, LLMService, CSV Pipeline, database Pipeline, or benchmark evaluation.

### Task 1: Make Evidence retrieval optional in the classification service

**Files:**
- Modify: `tests/unit/test_services.py`
- Modify: `app/services/classification_service.py`

- [ ] **Step 1: Write the failing no-RAG service test**

Add a test that fails if either Query construction or vector retrieval occurs and compares the actual LLM messages with the existing Prompt builder:

```python
from app.rag.prompt import build_classification_prompt


def test_classification_service_without_rag_skips_search_and_uses_empty_evidence():
    class ForbiddenStore:
        def search(self, query: str, k: int = 3):
            raise AssertionError("VectorStore.search must not run")

    class ForbiddenBuilder:
        def build(self, field: FieldProfile) -> str:
            raise AssertionError("retrieval Query must not be built")

    llm = FakeLanguageModel(
        ClassificationOutput(
            is_personal=True,
            category="个人基本资料",
            subcategory="姓名",
            level="L2",
            confidence=0.8,
            reason="字段画像显示为姓名。",
            need_review=False,
        )
    )
    profile = FieldProfile(
        source_system="csv",
        database_name="csv_source",
        table_name="catalog_input",
        field_name="customer_name",
        field_cn="客户姓名",
        field_comment="登记姓名",
        data_type="varchar",
        sample_values=["张三", "李四"],
        business_domain="customer",
    )
    service = FieldClassificationService(
        ForbiddenStore(),
        llm,
        query_builder=ForbiddenBuilder(),
        use_rag=False,
    )

    result = service.classify_field(profile)

    assert llm.prompt == build_classification_prompt(profile, [])
    assert '"field_cn": "客户姓名"' in llm.prompt[1].content
    assert "【检索依据（不可信数据）】\n[]" in llm.prompt[1].content
    assert result.is_personal is True
    assert result.evidence == []
    assert result.decision_path == "rag_llm"
```

- [ ] **Step 2: Run the new test and verify the interface is absent**

Run:

```powershell
python -m pytest tests/unit/test_services.py::test_classification_service_without_rag_skips_search_and_uses_empty_evidence -q
```

Expected: FAIL because `FieldClassificationService.__init__()` does not accept `use_rag`.

- [ ] **Step 3: Implement the minimal service switch**

In `FieldClassificationService.__init__()`, add and store the default-on flag:

```python
use_rag: bool = True,
```

```python
self.use_rag = use_rag
```

At the start of the existing `try` block in `classify_field()`, replace unconditional retrieval with:

```python
evidence = []
if self.use_rag:
    evidence = self.vector_store.search(self.build_query_text(profile), k=3)
    if not evidence:
        raise RuntimeError("知识库未检索到可用依据")
```

Leave the following call and all result/error mapping unchanged:

```python
output = self.llm_service.classify(
    build_classification_prompt(profile, evidence)
)
```

- [ ] **Step 4: Run service tests**

Run:

```powershell
python -m pytest tests/unit/test_services.py -q
```

Expected: all tests PASS, including existing retrieval tests proving default `use_rag=True` still calls `search(k=3)`.

- [ ] **Step 5: Run focused lint and commit**

Run:

```powershell
ruff check app/services/classification_service.py tests/unit/test_services.py
git diff --check
```

Expected: both commands PASS.

Commit:

```powershell
git add app/services/classification_service.py tests/unit/test_services.py
git commit -m "feat: add no-rag classification path"
```

### Task 2: Expose experiment D through the CSV CLI only

**Files:**
- Modify: `tests/integration/test_csv_pipeline_cli.py`
- Modify: `scripts/run_csv_pipeline.py`

- [ ] **Step 1: Write failing CLI parsing tests**

Add tests for the default, both explicit flags, and invalid strategy combination:

```python
@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--use-rag", True), ("--no-use-rag", False)],
)
def test_cli_accepts_rag_switch(tmp_path, flag: str, expected: bool):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(
        ["--input", str(path), "--query-strategy", "clean", flag]
    )

    assert args.use_rag is expected


def test_cli_enables_rag_by_default(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).use_rag is True


def test_cli_rejects_no_rag_for_non_clean_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--query-strategy",
                "profile",
                "--no-use-rag",
            ]
        )
```

- [ ] **Step 2: Extend the pipeline-construction test before implementation**

Update `CapturingClassifier` test doubles to accept `use_rag`, call:

```python
csv_cli.build_pipeline(
    settings,
    query_strategy="clean",
    use_rag=False,
)
```

and assert the captured keyword arguments include:

```python
{
    "query_strategy": "clean",
    "profile_query_mode": "c",
    "value_profiler": None,
    "use_rag": False,
}
```

Update the existing default rule-profiler assertion to include `"use_rag": True`; this proves existing CLI behavior stays enabled.

- [ ] **Step 3: Run CLI tests and verify they fail for the missing switch**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py -q
```

Expected: new tests FAIL because `--use-rag/--no-use-rag` and `build_pipeline(..., use_rag=...)` do not exist.

- [ ] **Step 4: Add the CLI flag and validation**

In `build_parser()`, add:

```python
parser.add_argument(
    "--use-rag",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="include retrieved Evidence in final classification",
)
```

In `parse_args()`, add:

```python
if not args.use_rag and args.query_strategy != "clean":
    parser.error("--no-use-rag requires --query-strategy clean")
```

- [ ] **Step 5: Propagate the switch without changing other pipelines**

Extend only this script's `build_pipeline()` signature:

```python
use_rag: bool = True,
```

Pass the flag to the existing classifier:

```python
classifier = FieldClassificationService(
    vector_store,
    LLMService(settings=settings),
    query_strategy=query_strategy,
    profile_query_mode=profile_query_mode,
    value_profiler=value_profiler,
    use_rag=use_rag,
)
```

Pass the parsed value from `main()` using a keyword to avoid changing existing positional meanings:

```python
pipeline = build_pipeline(
    load_settings(),
    args.query_strategy,
    args.query_mode,
    args.profiling_mode,
    use_rag=args.use_rag,
)
```

Do not change `scripts/run_benchmark.py`, `app/main.py`, or database Pipeline construction; their service default remains `True`.

- [ ] **Step 6: Run CLI and CSV Pipeline tests**

Run:

```powershell
python -m pytest tests/integration/test_csv_pipeline_cli.py tests/unit/test_csv_pipeline.py tests/integration/test_direct_csv_smoke.py -q
ruff check scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
```

Expected: all tests and Ruff PASS.

- [ ] **Step 7: Commit the CLI experiment switch**

```powershell
git add scripts/run_csv_pipeline.py tests/integration/test_csv_pipeline_cli.py
git commit -m "feat: expose no-rag evidence ablation"
```

### Task 3: Verify experimental isolation and complete the branch

**Files:**
- Verify only; no planned production changes

- [ ] **Step 1: Verify CLI help and argument combinations**

Run:

```powershell
python -m scripts.run_csv_pipeline --help
```

Expected: help includes both `--use-rag` and `--no-use-rag`.

Run a parse-only check:

```powershell
python -c "from scripts.run_csv_pipeline import parse_args; b=parse_args(['--input','RAG_mini_benchmark_150.csv','--query-strategy','clean','--use-rag']); d=parse_args(['--input','RAG_mini_benchmark_150.csv','--query-strategy','clean','--no-use-rag']); print(b.use_rag, d.use_rag)"
```

Expected output:

```text
True False
```

- [ ] **Step 2: Run the complete verification suite**

Run:

```powershell
python -m pytest -q
ruff check app scripts tests
python -m compileall -q app scripts
git diff --check
```

Expected: pytest, Ruff, compileall, and diff checks all PASS.

- [ ] **Step 3: Audit the modification boundary**

Run:

```powershell
git diff --name-status 8802a9d..HEAD
git diff 8802a9d..HEAD -- app/rag/prompt.py app/repositories/vector_store.py app/services/llm_service.py app/services/csv_pipeline.py app/services/database_pipeline.py app/services/benchmark_evaluator.py scripts/run_benchmark.py
```

Expected:

- The name list contains the implementation-plan document, the two implementation
  files, and their two test files.
- The second command has no output.

- [ ] **Step 4: Record the final experiment commands**

B command:

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --input-mode catalog --label-column expected_personal --query-strategy clean --use-rag
```

D command:

```powershell
python -m scripts.run_csv_pipeline --input "RAG_mini_benchmark_150.csv" --input-mode catalog --label-column expected_personal --query-strategy clean --no-use-rag
```

Do not run the paid 150-case experiments during implementation verification. Hand the commands to the user so both runs use their intended model, data, and collection configuration.
