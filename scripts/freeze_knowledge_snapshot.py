"""Freeze existing extraction artifacts without calling OCR or a language model."""

import argparse
import hashlib
import json
from pathlib import Path

from app.rag.chunker import split_knowledge_text
from app.schemas.knowledge_quality import ExtractedPage
from app.services.knowledge_cleaner import KnowledgeCleaner


def freeze(report_dir: Path, output_dir: Path, version: str) -> dict:
    if output_dir.exists():
        raise ValueError("snapshot directory already exists; never overwrite a freeze")
    documents = []
    sources = []
    for path in sorted(report_dir.glob("*.quality.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] == "FAIL":
            raise ValueError(f"failed source: {report['document_name']}")
        text_path = path.with_name(path.name.replace(".quality.json", ".cleaned.txt"))
        text = text_path.read_text(encoding="utf-8")
        cleaned = KnowledgeCleaner().clean(
            [ExtractedPage(page_number=1, text=text, extraction_method="text")]
        )
        name = report["document_name"]
        chunks = split_knowledge_text(
            cleaned.text, name,
            source_type="classification_rule" if name.endswith(".md") else "legal_document",
            source_format=Path(name).suffix.removeprefix("."), version=version,
            source_sha256=report["source_sha256"],
        )
        for chunk in chunks:
            chunk.metadata["extraction_quality_status"] = report["status"]
        documents.extend(chunks)
        sources.append({
            "document_name": name, "source_sha256": report["source_sha256"],
            "extraction_quality_status": report["status"], "issues": report["issues"],
            "input_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "cleaned_text": cleaned.text,
            "cleanup_audits": [audit.model_dump() for audit in cleaned.audits],
            "chunk_count": len(chunks),
        })
    if not sources:
        raise ValueError("no extraction artifacts found")
    payload = "".join(json.dumps({"content": d.page_content, "metadata": d.metadata},
                                ensure_ascii=False, sort_keys=True) + "\n" for d in documents)
    manifest = {
        "version": version, "query_strategy": "clean", "regulation_top_k": 3,
        "semantic_bridge": False, "retain_short_chunks": True,
        "provenance": "existing cleaned extraction artifacts; original page ranges unavailable",
        "review_pending": [s["document_name"] for s in sources
                           if s["extraction_quality_status"] == "REVIEW"],
        "chunk_count": len(documents),
        "maximum_chunk_characters": max(len(d.page_content) for d in documents),
        "chunks_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "sources": sources,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "chunks.jsonl").write_text(payload, encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    manifest = freeze(args.report_dir, args.output_dir, args.version)
    print(json.dumps({k: manifest[k] for k in
                      ("version", "chunk_count", "chunks_sha256", "review_pending")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
