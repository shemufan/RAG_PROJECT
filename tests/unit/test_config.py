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
        "KNOWLEDGE_BASE_VERSION",
        "SOURCE_DATABASE_URL",
        "TARGET_DATABASE_URL",
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
        "KNOWLEDGE_BASE_VERSION=v-test\n"
        "SOURCE_DATABASE_URL=mysql+pymysql://source/enterprise_source\n"
        "TARGET_DATABASE_URL=mysql+pymysql://target/compliance_result\n",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)

    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_timeout_seconds == 15
    assert settings.deepseek_max_retries == 1
    assert settings.embedding_model_path == tmp_path / "models" / "embedding"
    assert settings.chroma_db_dir == tmp_path / ".runtime" / "chroma"
    assert settings.knowledge_dir == tmp_path / "data" / "knowledge"
    assert settings.source_database_url.endswith("/enterprise_source")
    assert settings.target_database_url.endswith("/compliance_result")
