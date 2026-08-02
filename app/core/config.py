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
    knowledge_base_version: str
    knowledge_dir: Path


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
        knowledge_base_version=os.getenv("KNOWLEDGE_BASE_VERSION", "v1"),
        knowledge_dir=root / "data" / "knowledge",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(PROJECT_ROOT)
