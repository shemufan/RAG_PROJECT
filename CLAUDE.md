# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Gradio Web UI (development/demo)
python run_gradio.py                     # http://127.0.0.1:7860

# FastAPI backend (production/API)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# Swagger UI at http://127.0.0.1:8000/docs

# Code formatting
black . --line-length=100                # Python 3.10 target

# Install dependencies
pip install -r requirements.txt
```

## Architecture

This is a RAG-based enterprise data compliance classification system. It takes database field metadata (name, comment, sample values) and returns sensitivity levels L1–L4 with retrievable legal evidence from Chinese data protection laws.

### Critical initialization path (startup order)

Services MUST be initialized in this order — each depends on the previous:

```
EmbeddingService (HuggingFaceEmbeddings, local files only)
    ↓
ChromaStore (persistent vector DB at db/)
    ↓
LLMService (DeepSeek ChatOpenAI + structured output via function_calling)
    ↓
RAGClassifier(chroma_store, llm_service, error_cache)
```

This happens in two places with identical logic:
- `backend/main.py` lifespan (FastAPI) — stores singletons in `app.state` + module-level globals
- `run_gradio.py` main() (Gradio) — manual init, injects into frontend via `frontend.app.init_services()`

`get_services()` in `backend/main.py` is the bridge: Gradio imports it to get already-initialized services when both are running together.

### RAG inference pipeline (classify_field)

```
1. ERROR LOG CACHE CHECK  →  if field_name in cache: return cached level (cost=0, confidence=1.0)
2. BUILD SEARCH QUERY      →  concatenates field metadata into semantic query text
3. VECTOR RETRIEVAL (Top-3)→  ChromaStore.similarity_search() → cosine similarity
4. PROMPT ASSEMBLY         →  CLASSIFICATION_USER template with field_profile JSON + evidence JSON
5. STRUCTURED LLM CALL     →  LLMService.classify() → with_structured_output(method="function_calling")
6. RESULT WRAPPING          →  ClassificationResult with evidence, decision_path="rag_llm"
7. ON ERROR                 →  level="未知", decision_path="rag_llm_error"
```

### Hybrid rule + AI architecture

The system uses a **rules-first, AI-fallback** pattern:
- **Error log cache** (`RAGClassifier.error_cache`): dict of `{field_name: level}`. When a field name matches, the system returns immediately with `decision_path="error_log_cache"`, skipping RAG+LLM entirely. This is the "错题本" (error logbook) mechanism.
- **AI path**: The full RAG pipeline only runs when the field name is NOT in the cache.
- **Closed loop**: batch evaluation → human review of mismatches → export error log → re-upload → cache updated.

### Key design patterns

**`backend/core/config.py`** uses `os.path.dirname` x3 to resolve `BASE_DIR` to the project root, because it lives at `backend/core/config.py` (3 levels deep). All other paths (DB_PATH, DATA_DIR, etc.) derive from BASE_DIR. Environment overrides are loaded from `api_key.env` via `load_dotenv(override=True)`.

**All MySQL operations are graceful-failure**: `result_store.py` wraps every DB call in try/except, logging warnings but never raising. The Excel export path is the reliable fallback.

**Chroma vector store**: Embedded ChromaDB persisted at `db/` (project root). Embedding model is `sentence-transformers` MPNet 768-dim, loaded with `local_files_only=True` from `models/sentence-transformer/`. Documents are chunked by chapter/article boundaries via `split_by_article()` in `backend/utils/chunker.py`, not by character-count sliding windows.

**Structured output**: Uses `ChatOpenAI.with_structured_output(ClassificationOutput, method="function_calling")` for guaranteed JSON output — no manual regex fallback needed.

### Module boundaries (for team collaboration)

| Module | Key files | Must NOT modify |
|--------|-----------|-----------------|
| Backend API | `backend/main.py`, `backend/api/`, `backend/schemas/` | `backend/services/`, `frontend/` |
| RAG/LLM | `backend/services/` (rag_classifier, embedding_service, llm_service) | `backend/api/`, `frontend/` |
| Knowledge base | `backend/storage/chroma_store.py`, `backend/storage/mysql.py`, `data/` | `backend/api/`, `backend/services/rag_classifier.py` |
| Frontend | `frontend/app.py` | `backend/services/`, `backend/storage/`, `data/` |

When changing interface contracts (schemas, function signatures), update `docs/api_contract.md` and notify all members before merging.

### Git workflow

- `main` — stable releases only, never commit directly
- `dev` — integration branch for feature merges
- `feature/<module>` — per-member feature branches (e.g., `feature/backend-api`, `feature/rag-baseline`)
- PR must describe: what changed, which files, impact on other modules, how to test
- This repo is on branch `refactor/team-structure` (post-restructure, not yet merged)
