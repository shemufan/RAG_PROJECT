import json
from pathlib import Path
from types import SimpleNamespace

from scripts.rebuild_knowledge_base import execute, load_documents


def test_rebuild_script_loads_documents_from_configured_directory(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (tmp_path / "classification_rules.md").write_text(
        "### 规则 1：手机号\n- 等级: L3",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path, version="v1")

    assert len(documents) == 1
    assert "手机号" in documents[0].page_content


def settings_for(tmp_path: Path):
    return SimpleNamespace(
        knowledge_dir=tmp_path,
        knowledge_base_version="clean-v1",
        chroma_collection="data_classification",
        chroma_db_dir=tmp_path / "chroma",
        embedding_model_path=tmp_path / "model",
        qwen_ocr_api_key="",
        qwen_ocr_base_url="",
        qwen_ocr_model="qwen",
        qwen_ocr_cache_dir=tmp_path / "cache",
        qwen_ocr_timeout_seconds=10,
        qwen_ocr_max_retries=0,
    )


def test_check_only_writes_auditable_artifacts_without_vector_store(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "law.txt").write_text("第一条 文本法规", encoding="utf-8")
    report_root = tmp_path / "reports"
    vector_calls = []

    exit_code = execute(
        settings_for(tmp_path),
        check_only=True,
        report_dir=report_root,
        vector_store_factory=lambda *args, **kwargs: vector_calls.append(kwargs),
    )

    run_dir = report_root / "clean-v1"
    assert exit_code == 0
    assert vector_calls == []
    assert len(list(run_dir.glob("*.manifest.json"))) == 1
    assert len(list(run_dir.glob("*.quality.json"))) == 1
    assert len(list(run_dir.glob("*.cleaned.txt"))) == 1
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_count"] == 1
    assert summary["status_counts"]["PASS"] == 1


def test_check_only_returns_nonzero_for_failed_document(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "broken.txt").write_text("没有章节的文本", encoding="utf-8")

    exit_code = execute(
        settings_for(tmp_path),
        check_only=True,
        report_dir=tmp_path / "reports",
    )

    assert exit_code == 1


def test_rebuild_uses_versioned_candidate_collection_and_writes_import_report(tmp_path):
    laws = tmp_path / "laws"
    laws.mkdir()
    (laws / "law.txt").write_text("第一条 文本法规", encoding="utf-8")
    captured = {}

    class EmptyStore:
        def __init__(self):
            self.documents = []

        def count(self):
            return 0

        def add_documents(self, documents):
            self.documents.extend(documents)

    store = EmptyStore()

    def vector_store_factory(embedding_service, *, settings, collection_name):
        captured["collection_name"] = collection_name
        return store

    exit_code = execute(
        settings_for(tmp_path),
        report_dir=tmp_path / "reports",
        vector_store_factory=vector_store_factory,
        embedding_service_factory=lambda **kwargs: object(),
    )

    assert exit_code == 0
    assert captured["collection_name"] == "data_classification__clean-v1"
    report = json.loads(
        (tmp_path / "reports" / "clean-v1" / "import.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["chunk_count"] == 1
    assert report["maximum_chunk_characters"] <= 2000
