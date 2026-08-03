# Qwen PDF OCR Knowledge Ingestion Design

## Goal

Allow the existing knowledge-base rebuild command to ingest both UTF-8 text laws and PDF laws without manual conversion. PDF content is extracted through Alibaba Cloud Model Studio Qwen OCR, then passed through the existing legal chunking, embedding, and Chroma persistence flow.

The user workflow remains:

```text
copy TXT/PDF into data/knowledge/laws
-> python -m scripts.rebuild_knowledge_base
-> restart FastAPI
```

## Scope

This change includes:

- native discovery of `*.pdf` law documents;
- PDF validation before external upload;
- Qwen `qwen3.5-ocr` document parsing through the Beijing workspace API;
- temporary Model Studio upload for local PDF files;
- content-addressed local OCR caching;
- reuse of the existing legal chunker, embedding model, and Chroma repository;
- deterministic failure behavior and mock-based tests.

This change does not include:

- a document-upload HTTP endpoint or frontend;
- background jobs, queues, or parallel OCR;
- local OCR engines;
- permanent production OSS management;
- OCR for formats other than PDF;
- partial knowledge-base replacement after an extraction failure.

## Architecture

```text
classification_rules.md -----> text reader ---------+
laws/*.txt ------------------> text reader ---------+ |
                                                     v
laws/*.pdf -> PDF validator -> OCR cache -> Qwen OCR -> normalized text
                                                     |
                                                     v
                                      split_knowledge_text
                                                     |
                                                     v
                                      Embedding -> Chroma
```

### `QwenOCRService`

Add `app/services/ocr_service.py` with one public operation:

```python
extract_pdf(path: Path) -> str
```

Its responsibilities are limited to:

1. validate a local PDF;
2. calculate a cache key;
3. return a valid cached extraction when present;
4. upload a cache miss to Model Studio temporary storage;
5. call the Qwen Responses API with `qwen3.5-ocr` and `document_parsing`;
6. validate and cache the extracted text;
7. raise a typed exception containing the document name and safe failure reason.

It must not split documents, create embeddings, modify Chroma, or write tracked TXT files.

### Knowledge loader

`app/services/knowledge_service.py` continues to own document discovery and conversion into LangChain documents.

- `classification_rules.md` keeps its current text path.
- `laws/*.txt` keeps its current text path.
- `laws/*.pdf` is routed to the injected OCR service.
- TXT and PDF sources both enter `split_knowledge_text`.
- The original file name remains `document_name` so evidence points to the PDF.
- Metadata records `source_format` as `txt`, `md`, or `pdf`.
- Discovery order is deterministic.

If PDF files exist but no OCR service/configuration is available, loading fails explicitly. TXT-only repositories do not require Qwen OCR configuration.

### Rebuild script

`scripts/rebuild_knowledge_base.py` constructs `QwenOCRService` lazily only when PDF laws are present and passes it into the loader.

All source loading and OCR must finish before `KnowledgeService.rebuild()` resets Chroma. Therefore an invalid PDF or failed API call leaves the previous Chroma collection unchanged.

## Qwen Integration

Use the Alibaba Cloud Model Studio Beijing workspace OpenAI-compatible Responses API:

- model: `qwen3.5-ocr` by default;
- task: `document_parsing`;
- input: temporary `oss://` URL for the local PDF;
- prompt: extract all text in original reading order, preserve headings/articles/paragraphs, do not summarize or invent text, and represent unreadable characters with `?`.

The local upload uses the official temporary-upload policy endpoint. Temporary URLs are used only during synchronous rebuilds and are not persisted as application data.

## Configuration

Add centralized settings and `.env.example` entries:

```dotenv
QWEN_OCR_API_KEY=
QWEN_OCR_BASE_URL=
QWEN_OCR_MODEL=qwen3.5-ocr
QWEN_OCR_CACHE_DIR=.runtime/ocr_cache
QWEN_OCR_TIMEOUT_SECONDS=180
QWEN_OCR_MAX_RETRIES=2
```

`QWEN_OCR_BASE_URL` must contain the user's Beijing workspace domain, for example:

```text
https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

No API key or workspace ID is committed.

## Validation and Limits

Before upload, each PDF must satisfy:

- valid PDF structure;
- not encrypted;
- at least one page;
- no more than 50 pages;
- no more than 100 MB.

An extraction must contain non-whitespace text. Empty model output is an error.

The loader rejects two law files with the same case-insensitive base name but different extensions, such as `law.txt` and `law.pdf`, to prevent duplicate/conflicting evidence.

## Cache

OCR text is stored below `.runtime/ocr_cache`, which is already ignored by Git.

The cache identity contains:

- SHA-256 of the PDF bytes;
- OCR model name;
- OCR prompt/schema version.

A cache file is written atomically only after successful non-empty extraction. Changing the PDF, model, or prompt version causes a cache miss. Cached text remains an internal runtime artifact and is never copied into `data/knowledge/laws`.

## Error Handling

Define a dedicated OCR exception hierarchy or one typed `OCRExtractionError` with safe messages.

The rebuild aborts on:

- missing Qwen configuration when PDFs exist;
- invalid, encrypted, empty, oversized, or over-page-limit PDFs;
- upload failures;
- Qwen authentication, timeout, rate-limit, or exhausted-retry failures;
- malformed or empty OCR responses;
- conflicting TXT/PDF law names.

Logs may contain the file name, request ID, and error category, but never API keys, authorization headers, full signed URLs, or raw credentials.

No failed PDF is silently skipped, and no partial new collection replaces the previous collection.

## Dependencies

Add direct runtime dependencies because the project imports them directly:

- `openai` for the Responses API;
- `requests` for the official temporary-upload flow;
- `pypdf` for validation and page-count/encryption checks.

No Poppler, Tesseract, CUDA OCR model, or PDF-to-image dependency is introduced.

## Tests

All default tests use fakes/mocks and never call Model Studio.

Required coverage:

- existing TXT/Markdown loading remains unchanged;
- PDF discovery routes through an injected fake OCR service;
- OCR text reaches the existing legal chunker;
- PDF evidence keeps the original file name and `source_format=pdf`;
- cache hit avoids upload and API calls;
- cache miss calls OCR once and writes a valid cache;
- PDF/model/prompt changes invalidate the cache;
- corrupt, encrypted, empty, oversized, and over-limit PDFs fail explicitly;
- missing settings fail only when PDF input exists;
- empty/malformed API output fails;
- API failure does not reset the vector store;
- duplicate TXT/PDF base names fail;
- secrets and signed URLs are absent from errors.

An opt-in integration test may call Qwen using a small non-sensitive PDF. It is skipped unless dedicated test environment variables are present and must be reported separately from mock coverage.

## Operational Flow

After implementation:

1. place the teacher-provided PDFs in `data/knowledge/laws/`;
2. remove superseded law files to avoid mixed standards;
3. configure Qwen OCR settings in `.env`;
4. increment `KNOWLEDGE_BASE_VERSION`;
5. stop FastAPI;
6. run `python -m scripts.rebuild_knowledge_base`;
7. restart FastAPI;
8. classify a known field and confirm evidence references the new PDF file names.

## Security and Production Boundary

Every PDF processed by this path is uploaded to Alibaba Cloud Model Studio. Operators must ensure the document is permitted to leave the local machine.

Temporary Model Studio upload is appropriate for the current controlled synchronous rebuild workflow. A production ingestion service should replace that uploader with managed OSS, access control, retention policy, audit logging, and asynchronous job management without changing the `QwenOCRService.extract_pdf()` boundary.
