"""Opt-in retrieval comparison for current and candidate knowledge collections."""

import json
import os
from pathlib import Path

import pytest

from app.core.config import load_settings
from app.repositories.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService

RUN_FLAG = "RUN_KNOWLEDGE_RETRIEVAL_REGRESSION"
CANDIDATE_COLLECTION_ENV = "KNOWLEDGE_CANDIDATE_COLLECTION"
QUERIES = (
    ("个人信息的定义和识别范围是什么", ("35273", "45574", "41391")),
    ("敏感个人信息的定义及处理要求", ("45574", "35273")),
    ("网络数据分类分级的原则和方法", ("TC260",)),
    ("App 收集个人信息应遵循最小必要原则", ("41391",)),
    ("个人信息主体享有哪些访问更正删除权利", ("35273", "45574")),
    ("App 通讯录权限对应的必要个人信息范围", ("41391",)),
)


def _serialized_results(store: VectorStore, query: str) -> list[dict]:
    return [
        {
            "source": evidence.source,
            "article": evidence.article,
            "score": evidence.score,
            "chunk_id": evidence.chunk_id,
        }
        for evidence in store.search(query, k=3)
    ]


def _contains_expected_source(results: list[dict], expected: tuple[str, ...]) -> bool:
    return any(
        marker.lower() in result["source"].lower()
        for result in results
        for marker in expected
    )


def test_candidate_retrieval_keeps_core_sources_in_top_three():
    if os.getenv(RUN_FLAG) != "1":
        pytest.skip(f"{RUN_FLAG}=1 is required for the local Chroma regression")

    settings = load_settings(Path.cwd())
    candidate_collection = os.getenv(
        CANDIDATE_COLLECTION_ENV,
        f"{settings.chroma_collection}__{settings.knowledge_base_version}",
    )
    embedding_service = EmbeddingService(model_path=settings.embedding_model_path)
    current = VectorStore(
        client=None,
        embedding_service=embedding_service,
        settings=settings,
        collection_name=settings.chroma_collection,
    )
    candidate = VectorStore(
        client=None,
        embedding_service=embedding_service,
        settings=settings,
        collection_name=candidate_collection,
    )
    assert current.count() > 0, "current knowledge collection is empty"
    assert candidate.count() > 0, "candidate knowledge collection is empty"

    comparisons = []
    for query, expected_sources in QUERIES:
        current_results = _serialized_results(current, query)
        candidate_results = _serialized_results(candidate, query)
        assert _contains_expected_source(candidate_results, expected_sources), (
            f"candidate top-3 missed {expected_sources!r} for query: {query}"
        )
        comparisons.append(
            {
                "query": query,
                "expected_source_markers": list(expected_sources),
                "current": current_results,
                "candidate": candidate_results,
            }
        )

    report_path = (
        settings.project_root
        / ".runtime"
        / "knowledge_quality"
        / settings.knowledge_base_version
        / "retrieval-comparison.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
