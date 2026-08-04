# PDF Page Image OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace temporary OSS PDF input with locally rendered page images sent to Qwen OCR as Base64, while retaining inspectable page images outside Git.

**Architecture:** A focused `PdfImageService` renders and reuses page JPEGs beside each PDF. `QwenOCRService` remains the public PDF extraction boundary, checks its text cache first, then OCRs rendered pages through OpenAI-compatible Chat Completions and atomically caches the ordered result.

**Tech Stack:** Python 3.10, pypdf, pypdfium2, Pillow, OpenAI Python SDK, pytest, ruff

---

### Task 1: Add PDF page renderer through TDD

**Files:**
- Create: `app/services/pdf_image_service.py`
- Create: `tests/unit/test_pdf_image_service.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing renderer tests**

Add tests that create real blank PDFs with `PdfWriter` and assert:

```python
service = PdfImageService(dpi=200, jpeg_quality=90)
pages = service.render_pages(pdf_path)
assert [page.name for page in pages] == ["page-0001.jpg", "page-0002.jpg"]
assert all(page.parent == pdf_path.with_suffix("") for page in pages)
```

Also assert that a second call does not rewrite valid images, and changing the PDF page count invalidates the manifest and replaces the generated page set.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest -q tests/unit/test_pdf_image_service.py
```

Expected: collection fails because `app.services.pdf_image_service` does not exist.

- [ ] **Step 3: Add dependencies and precise ignore rules**

Add runtime dependencies:

```text
pypdfium2
Pillow
```

Add exact generated-file rules:

```gitignore
data/knowledge/laws/*/page-*.jpg
data/knowledge/laws/*/.pdf-pages.json
```

- [ ] **Step 4: Implement `PdfImageService` minimally**

Provide:

```python
class PdfImageService:
    def __init__(self, *, dpi: int = 200, jpeg_quality: int = 90): ...
    def render_pages(self, pdf_path: str | Path) -> list[Path]: ...
```

Use PDF SHA-256 plus render settings in `.pdf-pages.json`. Render to temporary JPEG names and replace atomically. Only delete files matching `page-[0-9][0-9][0-9][0-9].jpg` and the service manifest inside `pdf_path.with_suffix("")`.

- [ ] **Step 5: Run renderer tests and verify GREEN**

```powershell
pytest -q tests/unit/test_pdf_image_service.py
```

Expected: all renderer tests pass.

### Task 2: Replace OSS transport with Base64 page OCR through TDD

**Files:**
- Modify: `app/services/ocr_service.py`
- Modify: `tests/unit/test_ocr_service.py`

- [ ] **Step 1: Replace transport tests with desired Chat Completions behavior**

Create a fake page renderer and fake client shaped as `client.chat.completions.create`. Assert the request contains:

```python
{
    "model": "qwen3.5-ocr",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": LEGAL_OCR_PROMPT},
        ],
    }],
}
```

Assert two pages return ordered text containing explicit page boundaries, cache hits bypass both rendering and API calls, a page error names the one-based page number, and no final text cache is written after partial failure.

- [ ] **Step 2: Run OCR tests and verify RED**

```powershell
pytest -q tests/unit/test_ocr_service.py
```

Expected: failures show the service still expects HTTP upload and Responses API collaborators.

- [ ] **Step 3: Implement the Base64 Chat Completions path**

Remove `requests`, upload-policy constants, `_upload_pdf`, `_call_qwen` for `input_file`, and OSS-specific constructor arguments. Inject `image_service` and optional `chat_client`. For each rendered page:

```python
encoded = base64.b64encode(page.read_bytes()).decode("ascii")
data_url = f"data:image/jpeg;base64,{encoded}"
response = chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": [...] }],
)
text = response.choices[0].message.content
```

Reject an encoded payload over 10 MiB, empty output, and known missing-document refusals. Merge pages as `【第 N 页】\n{text}`. Preserve atomic final-cache writes and sanitized exception messages.

- [ ] **Step 4: Run OCR tests and verify GREEN**

```powershell
pytest -q tests/unit/test_ocr_service.py
```

Expected: all OCR unit tests pass with no network calls.

### Task 3: Wire the renderer and update integration coverage

**Files:**
- Modify: `scripts/rebuild_knowledge_base.py`
- Modify: `tests/unit/test_rebuild_pdf.py`
- Modify: `tests/integration/test_qwen_ocr_integration.py`

- [ ] **Step 1: Add failing construction assertions**

Assert `build_ocr_service()` returns a service using `PdfImageService`, and update the opt-in integration test to construct the new service without OSS collaborators.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
pytest -q tests/unit/test_rebuild_pdf.py tests/integration/test_qwen_ocr_integration.py
```

Expected: construction assertion fails until the renderer is wired; the paid integration test remains skipped unless explicitly enabled.

- [ ] **Step 3: Wire `PdfImageService`**

Construct the renderer in `build_ocr_service()` and inject it into `QwenOCRService`. Keep the existing PDF-only lazy construction and pre-Chroma load safety behavior.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
pytest -q tests/unit/test_rebuild_pdf.py tests/unit/test_knowledge_pdf_loading.py tests/integration/test_qwen_ocr_integration.py -rs
```

Expected: unit tests pass and the paid test reports skipped without `RUN_QWEN_OCR_INTEGRATION=1`.

### Task 4: Update operator documentation and verify the whole repository

**Files:**
- Modify: `README.md`
- Modify: `.env.example` only if descriptions require clarification
- Modify: `requirements.txt`

- [ ] **Step 1: Rewrite the PDF rebuild documentation**

Document the real flow: PDF validation, local same-name page directory, Base64 page OCR, ordered merge, text caching, and local/Git behavior. Remove all claims about temporary Alibaba storage, `oss://`, and OSS production migration.

- [ ] **Step 2: Confirm obsolete imports and terms are absent**

```powershell
Get-ChildItem app,scripts,tests -Recurse -File -Filter *.py | Select-String -Pattern 'import requests|UPLOAD_POLICY_URL|responses_client|http_session|oss://'
```

Expected: no production references; test fixtures also contain no obsolete transport implementation.

- [ ] **Step 3: Run complete verification**

```powershell
python -m compileall app scripts
ruff check .
pytest -q -rs
```

Expected: compile and ruff exit 0; all non-paid tests pass; external integration tests are explicitly skipped.

- [ ] **Step 4: Review the final diff and generated files**

```powershell
git diff --check
git status --short
```

Confirm no `page-*.jpg`, `.pdf-pages.json`, `.runtime` files, secrets, or unrelated user files are staged or overwritten.
