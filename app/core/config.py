"""Environment-backed settings loaded from the project root."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_timeout_seconds: float
    deepseek_max_retries: int
    embedding_model_path: Path | None
    chroma_db_dir: Path
    chroma_collection: str
    semantic_chroma_db_dir: Path
    semantic_chroma_collection: str
    semantic_knowledge_file: Path
    knowledge_base_version: str
    knowledge_dir: Path
    source_database_url: str | None
    target_database_url: str | None
    qwen_ocr_api_key: str
    qwen_ocr_base_url: str
    qwen_ocr_model: str
    qwen_ocr_cache_dir: Path
    qwen_ocr_timeout_seconds: float
    qwen_ocr_max_retries: int


def load_settings(project_root: str | Path = PROJECT_ROOT) -> Settings:
    """Load root ``.env`` values while allowing process variables to win."""
    root = Path(project_root).resolve()
    load_dotenv(root / ".env", override=False)
    embedding_value = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    return Settings(
        project_root=root,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1",
        ),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30")),
        deepseek_max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "2")),
        embedding_model_path=(
            _resolve_path(root, embedding_value) if embedding_value else None
        ),
        chroma_db_dir=_resolve_path(
            root,
            os.getenv("CHROMA_DB_DIR", ".runtime/chroma"),
        ),
        chroma_collection=os.getenv(
            "CHROMA_COLLECTION",
            "data_classification",
        ),
        semantic_chroma_db_dir=_resolve_path(
            root,
            os.getenv("SEMANTIC_CHROMA_DB_DIR", ".runtime/semantic_chroma"),
        ),
        semantic_chroma_collection=os.getenv(
            "SEMANTIC_CHROMA_COLLECTION",
            "semantic_field_types",
        ),
        semantic_knowledge_file=_resolve_path(
            root,
            os.getenv(
                "SEMANTIC_KNOWLEDGE_FILE",
                "data/semantic_knowledge/semantic_cards.json",
            ),
        ),
        knowledge_base_version=os.getenv("KNOWLEDGE_BASE_VERSION", "v1"),
        knowledge_dir=root / "data" / "knowledge",
        source_database_url=os.getenv("SOURCE_DATABASE_URL") or None,
        target_database_url=os.getenv("TARGET_DATABASE_URL") or None,
        qwen_ocr_api_key=os.getenv("QWEN_OCR_API_KEY", ""),
        qwen_ocr_base_url=os.getenv("QWEN_OCR_BASE_URL", ""),
        qwen_ocr_model=os.getenv("QWEN_OCR_MODEL", "qwen3.5-ocr"),
        qwen_ocr_cache_dir=_resolve_path(
            root,
            os.getenv("QWEN_OCR_CACHE_DIR", ".runtime/ocr_cache"),
        ),
        qwen_ocr_timeout_seconds=float(
            os.getenv("QWEN_OCR_TIMEOUT_SECONDS", "180")
        ),
        qwen_ocr_max_retries=int(os.getenv("QWEN_OCR_MAX_RETRIES", "2")),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(PROJECT_ROOT)
