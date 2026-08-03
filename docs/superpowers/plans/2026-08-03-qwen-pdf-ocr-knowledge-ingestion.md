# Qwen PDF OCR Knowledge Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing knowledge-base rebuild command ingest local PDF laws through Qwen OCR automatically, with validation, caching, atomic failure behavior, and no manual TXT conversion.

**Architecture:** Add a focused `QwenOCRService` that validates and content-addresses PDFs, uploads cache misses to Model Studio temporary storage, calls the Beijing workspace Responses API, and returns normalized text. Extend the existing knowledge loader to route `*.pdf` through the injected service, then reuse the current chunker, embedding service, and Chroma rebuild path. Load every source successfully before resetting Chroma.

**Tech Stack:** Python 3.10+, `pypdf`, `requests`, OpenAI Python SDK Responses API, LangChain `Document`, Chroma, pytest, Ruff.

---

## File Map

- Modify `app/core/config.py`, `.env.example`, `requirements.txt`: OCR configuration and runtime dependencies.
- Modify `app/services/ocr_service.py`: fill the user's empty placeholder with PDF validation, cache, temporary upload, and Qwen OCR.
- Modify `app/services/knowledge_service.py`, `app/rag/chunker.py`: discover and convert TXT/PDF laws with traceable metadata.
- Modify `scripts/rebuild_knowledge_base.py`: lazy OCR construction and atomic source loading.
- Create `tests/unit/test_ocr_service.py`: network-free validation, cache, and transport tests.
- Modify existing unit tests and create `tests/integration/test_qwen_ocr_integration.py`.
- Modify `README.md`: exact PDF ingestion operation and security boundary.

### Task 1: Centralize OCR configuration and dependencies

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing configuration test**

Extend the environment cleanup and temporary `.env` in `test_config_loads_project_root_env_and_resolves_relative_paths` with:

```python
"QWEN_OCR_API_KEY",
"QWEN_OCR_BASE_URL",
"QWEN_OCR_MODEL",
"QWEN_OCR_CACHE_DIR",
"QWEN_OCR_TIMEOUT_SECONDS",
"QWEN_OCR_MAX_RETRIES",
```

Write test values and assert:

```python
assert settings.qwen_ocr_api_key == "ocr-test-key"
assert settings.qwen_ocr_base_url == "https://workspace.example.test/compatible-mode/v1"
assert settings.qwen_ocr_model == "qwen3.5-ocr"
assert settings.qwen_ocr_cache_dir == tmp_path / ".runtime" / "ocr_cache"
assert settings.qwen_ocr_timeout_seconds == 90
assert settings.qwen_ocr_max_retries == 3
```

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m pytest tests/unit/test_config.py -q
```

Expected: FAIL because `Settings` has no Qwen OCR attributes.

- [ ] **Step 3: Add centralized settings and safe defaults**

Add fields and load them only in `app/core/config.py`:

```python
qwen_ocr_api_key: str
qwen_ocr_base_url: str
qwen_ocr_model: str
qwen_ocr_cache_dir: Path
qwen_ocr_timeout_seconds: float
qwen_ocr_max_retries: int
```

```python
qwen_ocr_api_key=os.getenv("QWEN_OCR_API_KEY", ""),
qwen_ocr_base_url=os.getenv("QWEN_OCR_BASE_URL", ""),
qwen_ocr_model=os.getenv("QWEN_OCR_MODEL", "qwen3.5-ocr"),
qwen_ocr_cache_dir=_resolve_path(root, os.getenv("QWEN_OCR_CACHE_DIR", ".runtime/ocr_cache")),
qwen_ocr_timeout_seconds=float(os.getenv("QWEN_OCR_TIMEOUT_SECONDS", "180")),
qwen_ocr_max_retries=int(os.getenv("QWEN_OCR_MAX_RETRIES", "2")),
```

Add matching entries to `.env.example`, without keys or workspace IDs. Add direct runtime dependencies:

```text
openai
requests
pypdf
```

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/unit/test_config.py -q
python -m ruff check app/core/config.py tests/unit/test_config.py
git add .env.example requirements.txt app/core/config.py tests/unit/test_config.py
git commit -m "feat: configure qwen pdf ocr"
```

### Task 2: Implement PDF validation and content-addressed cache

**Files:**
- Create: `tests/unit/test_ocr_service.py`
- Modify: `app/services/ocr_service.py`

- [ ] **Step 1: Write failing validation and cache tests**

Create PDFs using `pypdf.PdfWriter` and cover corrupt, encrypted, zero-page, over-50-page, cache-hit, cache-miss, changed-PDF, and changed-model behavior. Inject an extractor so no test uses the network:

```python
service = QwenOCRService(
    api_key="test",
    base_url="https://workspace.example.test/v1",
    model="qwen3.5-ocr",
    cache_dir=tmp_path / "cache",
    extractor=lambda path: "第一条 测试法规",
)
assert service.extract_pdf(pdf_path) == "第一条 测试法规"
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tests/unit/test_ocr_service.py -q
```

Expected: FAIL because the placeholder service is empty.

- [ ] **Step 3: Implement the stable public boundary**

```python
MAX_PDF_BYTES = 100 * 1024 * 1024
MAX_PDF_PAGES = 50
OCR_PROMPT_VERSION = "legal-document-v1"


class OCRExtractionError(RuntimeError):
    """Raised when a PDF cannot be safely converted to knowledge text."""


class QwenOCRService:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        cache_dir: Path,
        timeout_seconds: float = 180,
        max_retries: int = 2,
        extractor=None,
        http_session=None,
        responses_client=None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._extractor = extractor
        self._http_session = http_session
        self._responses_client = responses_client

    def extract_pdf(self, path: Path) -> str:
        pdf_path = Path(path)
        self._validate_pdf(pdf_path)
        cache_path = self._cache_path(pdf_path)
        if cache_path.is_file():
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        extractor = self._extractor or self._extract_with_qwen
        text = extractor(pdf_path).strip()
        if not text:
            raise OCRExtractionError(f"{pdf_path.name}: OCR returned empty text")
        self._write_cache(cache_path, text)
        return text
```

Use `PdfReader` to reject damaged/encrypted/invalid-page-count PDFs. Calculate SHA-256 over PDF bytes, model, and prompt version. Write non-empty UTF-8 cache content to a unique temporary sibling and atomically replace the target. Include the file name in safe errors, never credentials or URLs.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/unit/test_ocr_service.py -q
python -m ruff check app/services/ocr_service.py tests/unit/test_ocr_service.py
git add app/services/ocr_service.py tests/unit/test_ocr_service.py
git commit -m "feat: validate and cache pdf ocr text"
```

### Task 3: Add Model Studio upload and Qwen Responses call

**Files:**
- Modify: `tests/unit/test_ocr_service.py`
- Modify: `app/services/ocr_service.py`

- [ ] **Step 1: Write failing transport tests**

Use fake `requests.Session` responses and a fake OpenAI Responses client to cover successful upload/extraction, policy failure, upload failure, empty Qwen output, safe error text, and exact request structure. Assert:

```python
assert request["model"] == "qwen3.5-ocr"
assert request["extra_body"] == {"ocr_options": {"task": "document_parsing"}}
assert request["input"][0]["content"][0]["type"] == "input_file"
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest tests/unit/test_ocr_service.py -q
```

Expected: transport tests FAIL.

- [ ] **Step 3: Implement upload and OCR**

Implement private `_upload_pdf`, `_call_qwen`, and `_extract_with_qwen` methods. Upload flow:

1. GET `https://dashscope.aliyuncs.com/api/v1/uploads` with `action=getPolicy` and the configured model.
2. POST the returned multipart policy and PDF stream to `upload_host`.
3. Return `oss://<upload_dir>/<file-name>`.
4. Wrap failures without including response bodies, keys, or signed values.

Construct the production client lazily:

```python
OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=timeout_seconds,
    max_retries=max_retries,
)
```

Call `responses.create` with `document_parsing`, the `input_file`, and a user instruction to preserve titles, chapters, articles, and paragraphs without summarizing or inventing content. Normalize `response.output_text.strip()` and reject empty output.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/unit/test_ocr_service.py -q
python -m ruff check app/services/ocr_service.py tests/unit/test_ocr_service.py
git add app/services/ocr_service.py tests/unit/test_ocr_service.py
git commit -m "feat: extract pdf text with qwen ocr"
```

### Task 4: Route PDF laws through the knowledge loader

**Files:**
- Modify: `tests/unit/test_services.py`
- Modify: `tests/unit/test_rag.py`
- Modify: `app/services/knowledge_service.py`
- Modify: `app/rag/chunker.py`

- [ ] **Step 1: Write failing loader tests**

Inject a fake OCR service and cover PDF routing, original PDF document name, `source_format=pdf`, missing OCR service, and duplicate case-insensitive TXT/PDF stems:

```python
class FakeOCRService:
    def extract_pdf(self, path: Path) -> str:
        return "第一章 总则\n第一条 PDF法规内容"
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest tests/unit/test_services.py tests/unit/test_rag.py -q
```

Expected: PDF tests FAIL because only `*.txt` is discovered.

- [ ] **Step 3: Implement deterministic multi-format loading**

Extend the chunker without breaking callers:

```python
def split_knowledge_text(
    text: str,
    document_name: str,
    *,
    source_type: str,
    version: str,
    source_format: str = "txt",
) -> list[Document]:
```

Add `source_format` to chunk metadata. Extend `load_knowledge_documents(knowledge_dir, *, version, ocr_service=None)` to discover sorted TXT/PDF laws, reject duplicate stems, route PDFs to `extract_pdf`, and raise clearly when PDF input lacks OCR configuration.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/unit/test_services.py tests/unit/test_rag.py -q
python -m ruff check app/services/knowledge_service.py app/rag/chunker.py tests/unit/test_services.py tests/unit/test_rag.py
git add app/services/knowledge_service.py app/rag/chunker.py tests/unit/test_services.py tests/unit/test_rag.py
git commit -m "feat: load pdf laws through ocr"
```

### Task 5: Wire OCR into atomic knowledge rebuild

**Files:**
- Modify: `tests/unit/test_rebuild_script.py`
- Modify: `scripts/rebuild_knowledge_base.py`

- [ ] **Step 1: Write failing orchestration tests**

Cover TXT-only operation without OCR, PDF operation with injected OCR, missing settings, and OCR failure before vector reset. Use a store recording `reset()` and `add_documents()` and assert `reset_called is False` after OCR failure.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest tests/unit/test_rebuild_script.py -q
```

Expected: PDF injection/orchestration tests FAIL.

- [ ] **Step 3: Add lazy testable wiring**

Extend `load_documents(knowledge_dir, *, version, ocr_service=None)`. Add `build_ocr_service(settings)` that rejects missing key/base URL and constructs `QwenOCRService`. In `main`, construct OCR only if `laws/*.pdf` exists, load every document first, then call `KnowledgeService(store).rebuild(documents)`.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/unit/test_rebuild_script.py tests/unit/test_services.py tests/unit/test_ocr_service.py -q
python -m ruff check scripts/rebuild_knowledge_base.py tests/unit/test_rebuild_script.py
git add scripts/rebuild_knowledge_base.py tests/unit/test_rebuild_script.py
git commit -m "feat: rebuild knowledge from pdf laws"
```

### Task 6: Documentation, opt-in integration, and final verification

**Files:**
- Create: `tests/integration/test_qwen_ocr_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Add an opt-in real integration test**

Read only `QWEN_OCR_TEST_PDF`, `QWEN_OCR_API_KEY`, `QWEN_OCR_BASE_URL`, and optional `QWEN_OCR_MODEL`. Skip explicitly unless configured. Require an existing non-sensitive PDF below 5 MB, use a temporary cache, call `extract_pdf`, and assert non-empty output. Default pytest must never call Qwen.

- [ ] **Step 2: Document exact operation**

Document this workflow:

```text
copy permitted TXT/PDF laws to data/knowledge/laws
-> configure QWEN_OCR_* when PDFs exist
-> increase KNOWLEDGE_BASE_VERSION
-> stop FastAPI
-> python -m scripts.rebuild_knowledge_base
-> restart FastAPI
-> verify evidence document_name
```

Also document 50-page/100-MB limits, `.runtime/ocr_cache`, cloud-upload/privacy boundary, atomic failure behavior, and temporary-upload production limits.

- [ ] **Step 3: Install dependencies and run all verification**

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m compileall app scripts
python -m ruff check .
python -m pytest -q -rs
python -m pip check
```

Expected: compile/Ruff/mock tests pass; real MySQL and Qwen tests skip unless explicitly configured.

- [ ] **Step 4: Review secrets, residue, and diff**

```powershell
git diff --check 58711c5..HEAD
git grep -n -I -E "TODO|FIXME|DASHSCOPE_API_KEY=sk-|QWEN_OCR_API_KEY=sk-" -- app scripts tests README.md .env.example requirements.txt
git status --short --branch
```

Confirm TXT-only rebuild never requires Qwen, every failed PDF aborts before reset, cache identity covers PDF/model/prompt, errors do not expose credentials, and README names match code.

- [ ] **Step 5: Commit and report honestly**

```powershell
git add README.md tests/integration/test_qwen_ocr_integration.py
git commit -m "docs: document pdf ocr knowledge rebuild"
```

If review finds a defect, first add a reproducing test, fix it, rerun the entire verification set, and commit the correction separately. Report whether a paid Qwen request was actually observed; never claim real OCR success from mock coverage.
