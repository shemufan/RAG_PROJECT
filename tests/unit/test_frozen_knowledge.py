import hashlib
import json
from pathlib import Path

import pytest

from app.rag.chunker import split_knowledge_text
from app.schemas.field import FieldProfile
from app.services.classification_service import FieldClassificationService
from scripts.rebuild_frozen_knowledge import load_snapshot


def test_frozen_snapshot_preserves_short_chunks_and_excludes_directory_noise():
    root = Path(__file__).resolve().parents[2] / "data/knowledge_snapshots/b-rebuild-20260917"
    payload = (root / "chunks.jsonl").read_bytes()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(payload).hexdigest() == manifest["chunks_sha256"]
    chunks = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    assert len(chunks) == manifest["chunk_count"]
    assert any(len(chunk["content"]) < 20 for chunk in chunks)
    for chunk in chunks:
        for line in chunk["content"].splitlines():
            assert "".join(line.split()) not in {"目录", "目次", "-", "—", "---"}
            assert not (line.startswith("参考文献") and "……" in line)


def test_default_query_uses_only_field_name_and_samples():
    service = FieldClassificationService(None, None)
    query = service.build_query_text(FieldProfile(
        field_name="mobile", sample_values=["138****1234"],
        field_comment="private metadata", table_name="customers",
    ))
    assert query == "field_name: mobile\nsample_values: 138****1234"


def test_restore_keeps_source_review_gate(tmp_path):
    root = Path(__file__).resolve().parents[2] / "data/knowledge_snapshots/b-rebuild-20260917"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        source.pop("human_review", None)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_bytes((root / "chunks.jsonl").read_bytes())
    with pytest.raises(ValueError, match="source review required"):
        load_snapshot(tmp_path, set())
    approvals = {s["source_sha256"] for s in manifest["sources"]
                 if s["extraction_quality_status"] == "REVIEW"}
    restored, documents = load_snapshot(tmp_path, approvals)
    assert len(documents) == restored["chunk_count"]


def test_restore_accepts_persisted_human_review_bound_to_source_hash(tmp_path):
    root = Path(__file__).resolve().parents[2] / "data/knowledge_snapshots/b-rebuild-20260917"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        if source["extraction_quality_status"] == "REVIEW":
            source["human_review"] = {
                "status": "APPROVED", "source_sha256": source["source_sha256"]
            }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_bytes((root / "chunks.jsonl").read_bytes())
    assert len(load_snapshot(tmp_path, set())[1]) == manifest["chunk_count"]
    for source in manifest["sources"]:
        if source["extraction_quality_status"] == "REVIEW":
            source["human_review"]["source_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source review required"):
        load_snapshot(tmp_path, set())


def test_isolated_chapter_and_appendix_headings_are_chunks():
    documents = split_knowledge_text(
        "第一章 总则\n附录 A", "test.txt", source_type="legal_document", version="test"
    )
    assert [d.page_content for d in documents] == ["第一章 总则", "附录 A"]
