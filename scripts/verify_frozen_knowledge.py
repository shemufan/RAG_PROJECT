"""Verify the active frozen knowledge collection without calling a classification LLM."""

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.repositories.vector_store import VectorStore
from app.schemas.field import FieldProfile
from app.services.classification_service import FieldClassificationService
from app.services.embedding_service import EmbeddingService
from scripts.rebuild_frozen_knowledge import load_snapshot


def verify_active_configuration(settings, manifest) -> None:
    expected = f"data_classification__{manifest['version']}"
    if (
        settings.chroma_collection != expected
        or settings.knowledge_base_version != manifest["version"]
    ):
        raise ValueError("active collection/version differs from frozen snapshot")


def verify_collection(store, documents) -> None:
    if store.count() != len(documents):
        raise ValueError("collection count differs from snapshot")
    rows = store._store.get(include=["documents", "metadatas"])
    expected = {d.metadata["chunk_id"]: (d.page_content, d.metadata) for d in documents}
    actual = {
        metadata["chunk_id"]: (text, metadata)
        for metadata, text in zip(rows["metadatas"], rows["documents"], strict=True)
    }
    if len(expected) != len(documents) or actual != expected:
        raise ValueError("active vector collection content/metadata differs from frozen snapshot")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path,
        default=settings.project_root / "data/knowledge_snapshots/b-rebuild-20260917",
    )
    args = parser.parse_args()
    manifest, documents = load_snapshot(args.snapshot, set())
    verify_active_configuration(settings, manifest)
    store = VectorStore(
        EmbeddingService(model_path=settings.embedding_model_path), settings=settings,
    )
    verify_collection(store, documents)
    expected_ids = {d.metadata["chunk_id"] for d in documents}
    service = FieldClassificationService(store, None)
    probes = [
        ("mobile", ["138****1234"]), ("email", ["a****@example.test"]),
        ("id_card_no", ["3401**********1234"]),
        ("password_hash", ["synthetic_hash"]), ("product_name", ["office chair"]),
    ]
    results = []
    for name, samples in probes:
        query = service.build_query_text(FieldProfile(field_name=name, sample_values=samples))
        evidence = store.search(query, k=3)
        if len(evidence) != 3 or any(e.chunk_id not in expected_ids for e in evidence):
            raise ValueError(f"frozen Top-3 retrieval failed for {name}")
        results.append({"field_name": name, "query": query,
                        "top3": [e.model_dump() for e in evidence]})
    report = {
        "collection": settings.chroma_collection, "version": manifest["version"],
        "chunks": len(documents), "snapshot_match": True, "active_configuration_match": True,
        "queries": results,
    }
    output = settings.project_root / ".runtime/frozen_knowledge_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                     ("collection", "version", "chunks", "snapshot_match",
                      "active_configuration_match")}))
    print(f"Mainline Query retrieval probes passed: {len(results)}")


if __name__ == "__main__":
    main()
