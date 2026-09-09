from app.core.config import load_settings


def test_config_loads_project_root_env_and_resolves_relative_paths(tmp_path, monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
        "EMBEDDING_MODEL_PATH",
        "CHROMA_DB_DIR",
        "CHROMA_COLLECTION",
        "SEMANTIC_CHROMA_DB_DIR",
        "SEMANTIC_CHROMA_COLLECTION",
        "SEMANTIC_KNOWLEDGE_FILE",
        "KNOWLEDGE_BASE_VERSION",
        "SOURCE_DATABASE_URL",
        "TARGET_DATABASE_URL",
        "QWEN_OCR_API_KEY",
        "QWEN_OCR_BASE_URL",
        "QWEN_OCR_MODEL",
        "QWEN_OCR_CACHE_DIR",
        "QWEN_OCR_TIMEOUT_SECONDS",
        "QWEN_OCR_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=test-key\n"
        "DEEPSEEK_BASE_URL=https://example.test/v1\n"
        "DEEPSEEK_MODEL=test-model\n"
        "DEEPSEEK_TIMEOUT_SECONDS=15\n"
        "DEEPSEEK_MAX_RETRIES=1\n"
        "EMBEDDING_MODEL_PATH=models/embedding\n"
        "CHROMA_DB_DIR=.runtime/chroma\n"
        "CHROMA_COLLECTION=test_collection\n"
        "SEMANTIC_CHROMA_DB_DIR=.runtime/semantic-test\n"
        "SEMANTIC_CHROMA_COLLECTION=semantic_test_collection\n"
        "SEMANTIC_KNOWLEDGE_FILE=data/semantic_knowledge/test_cards.json\n"
        "KNOWLEDGE_BASE_VERSION=v-test\n"
        "SOURCE_DATABASE_URL=mysql+pymysql://source/enterprise_source\n"
        "TARGET_DATABASE_URL=mysql+pymysql://target/compliance_result\n"
        "QWEN_OCR_API_KEY=ocr-test-key\n"
        "QWEN_OCR_BASE_URL=https://workspace.example.test/compatible-mode/v1\n"
        "QWEN_OCR_MODEL=qwen3.5-ocr\n"
        "QWEN_OCR_CACHE_DIR=.runtime/ocr_cache\n"
        "QWEN_OCR_TIMEOUT_SECONDS=90\n"
        "QWEN_OCR_MAX_RETRIES=3\n",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)

    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_timeout_seconds == 15
    assert settings.deepseek_max_retries == 1
    assert settings.embedding_model_path == tmp_path / "models" / "embedding"
    assert settings.chroma_db_dir == tmp_path / ".runtime" / "chroma"
    assert settings.semantic_chroma_db_dir == tmp_path / ".runtime" / "semantic-test"
    assert settings.semantic_chroma_collection == "semantic_test_collection"
    assert settings.semantic_knowledge_file == (
        tmp_path / "data" / "semantic_knowledge" / "test_cards.json"
    )
    assert settings.semantic_chroma_db_dir != settings.chroma_db_dir
    assert settings.semantic_chroma_collection != settings.chroma_collection
    assert settings.knowledge_dir == tmp_path / "data" / "knowledge"
    assert settings.source_database_url.endswith("/enterprise_source")
    assert settings.target_database_url.endswith("/compliance_result")
    assert settings.qwen_ocr_api_key == "ocr-test-key"
    assert settings.qwen_ocr_base_url == "https://workspace.example.test/compatible-mode/v1"
    assert settings.qwen_ocr_model == "qwen3.5-ocr"
    assert settings.qwen_ocr_cache_dir == tmp_path / ".runtime" / "ocr_cache"
    assert settings.qwen_ocr_timeout_seconds == 90
    assert settings.qwen_ocr_max_retries == 3
