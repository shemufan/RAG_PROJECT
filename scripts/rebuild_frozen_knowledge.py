"""Restore a frozen chunk snapshot into a new candidate regulation collection."""

import argparse
import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document

from app.services.knowledge_service import KnowledgeService


def load_snapshot(root: Path, approved_hashes: set[str]) -> tuple[dict, list[Document]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    payload = (root / "chunks.jsonl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["chunks_sha256"]:
        raise ValueError("frozen snapshot checksum mismatch")
    for source in manifest["sources"]:
        if source["extraction_quality_status"] == "REVIEW":
            review = source.get("human_review", {})
            reviewed = (
                review.get("status") == "APPROVED"
                and review.get("source_sha256") == source["source_sha256"]
            )
            if not reviewed and source["source_sha256"] not in approved_hashes:
                raise ValueError(f"source review required: {source['document_name']}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(rows) != manifest["chunk_count"]:
        raise ValueError("frozen snapshot chunk count mismatch")
    return manifest, [Document(page_content=r["content"], metadata=r["metadata"]) for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--approve-review", action="append", default=[], metavar="SHA256")
    args = parser.parse_args()
    manifest, documents = load_snapshot(args.snapshot, set(args.approve_review))
    from app.core.config import get_settings
    from app.repositories.vector_store import VectorStore
    from app.services.embedding_service import EmbeddingService

    settings = get_settings()
    collection = f"data_classification__{manifest['version']}"
    if collection == settings.chroma_collection:
        raise ValueError("restore to an inactive candidate collection")
    store = VectorStore(EmbeddingService(model_path=settings.embedding_model_path),
                        settings=settings, collection_name=collection)
    count = KnowledgeService(store).rebuild(documents)
    print(f"Restored {count} chunks to {collection}; active collection unchanged")


if __name__ == "__main__":
    main()
