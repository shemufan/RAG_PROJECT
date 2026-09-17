"""Verify frozen vector content and exercise B Queries without calling a classification LLM."""

import json
from pathlib import Path

from app.core.config import get_settings
from app.repositories.vector_store import VectorStore
from app.services.classification_service import FieldClassificationService
from app.services.embedding_service import EmbeddingService
from scripts.rebuild_frozen_knowledge import load_snapshot


def main():
    root = Path("data/knowledge_snapshots/b-rebuild-20260917")
    manifest, documents = load_snapshot(root, set())
    settings = get_settings()
    collection = f"data_classification__{manifest['version']}"
    store = VectorStore(EmbeddingService(model_path=settings.embedding_model_path),
                        settings=settings, collection_name=collection)
    assert store.count() == len(documents), "collection count differs from snapshot"
    rows = store._store.get(include=["documents", "metadatas"])
    expected = {d.metadata["chunk_id"]: d.page_content for d in documents}
    actual = {metadata["chunk_id"]: text for metadata, text in
              zip(rows["metadatas"], rows["documents"])}
    assert actual == expected, "vector collection text differs from frozen snapshot"
    assert all(m["version"] == manifest["version"] for m in rows["metadatas"])
    service = FieldClassificationService(store, None)
    probes = [
        ("mobile", ["138****1234"]), ("email", ["a****@example.test"]),
        ("id_card_no", ["3401**********1234"]),
        ("password_hash", ["synthetic_hash"]), ("product_name", ["office chair"]),
    ]
    results = []
    from app.schemas.field import FieldProfile

    for name, samples in probes:
        query = service.build_query_text(FieldProfile(field_name=name, sample_values=samples))
        evidence = store.search(query, k=3)
        assert len(evidence) == 3
        assert all(e.chunk_id in expected for e in evidence)
        results.append({"field_name": name, "query": query,
                        "top3": [e.model_dump() for e in evidence]})
    report = {"collection": collection, "version": manifest["version"],
              "chunks": len(documents), "snapshot_match": True, "queries": results}
    output = settings.project_root / ".runtime/frozen_knowledge_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("collection", "version", "chunks", "snapshot_match")}))
    print(f"B Query retrieval probes passed: {len(results)}")


if __name__ == "__main__":
    main()
