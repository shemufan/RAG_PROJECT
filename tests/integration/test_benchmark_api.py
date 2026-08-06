from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.api.benchmark import get_benchmark_repository
from app.main import create_app
from app.schemas.benchmark import BenchmarkRunSummary

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(self):
        self.filters = None

    def get_run(self, run_id):
        if run_id != RUN_ID:
            return None
        return BenchmarkRunSummary(
            run_id=RUN_ID,
            batch_name="teacher_2026_08",
            status="SUCCESS",
            total_cases=10,
            success_cases=10,
            tp=4,
            fp=1,
            tn=5,
            fn=0,
            precision_score=0.8,
            recall_score=1.0,
            f1_score=0.88888889,
            accuracy_score=0.9,
            coverage_score=1.0,
            effective_recall_score=1.0,
            model_name="fake",
            knowledge_base_version="v-test",
            started_at=NOW,
            finished_at=NOW,
        )

    def query_predictions(self, **filters):
        self.filters = filters
        return [
            {
                "prediction_id": 1,
                "run_id": RUN_ID,
                "benchmark_id": 7,
                "field_name_snapshot": "email",
                "sample_values": ["masked"],
                "expected_personal": True,
                "predicted_personal": False,
                "outcome": "FN",
                "category": "non-personal",
                "level": "L1",
                "confidence": 0.7,
                "reason": "test",
                "need_review": True,
                "decision_path": "rag_llm",
                "evidence": [],
                "status": "SUCCESS",
                "created_at": NOW,
            }
        ]


@asynccontextmanager
async def empty_lifespan(app):
    yield


def make_client():
    repository = FakeRepository()
    application = create_app(lifespan=empty_lifespan)
    application.dependency_overrides[get_benchmark_repository] = lambda: repository
    return TestClient(application), repository


def test_benchmark_run_endpoint_returns_metrics_and_404():
    client, _ = make_client()
    with client:
        found = client.get(f"/api/benchmark/runs/{RUN_ID}")
        missing = client.get(f"/api/benchmark/runs/{uuid4()}")

    assert found.status_code == 200
    assert found.json()["precision_score"] == 0.8
    assert missing.status_code == 404


def test_benchmark_results_validate_and_forward_filters():
    client, repository = make_client()
    with client:
        invalid_outcome = client.get(
            "/api/benchmark/results",
            params={"run_id": str(RUN_ID), "outcome": "INVALID"},
        )
        invalid_limit = client.get(
            "/api/benchmark/results",
            params={"run_id": str(RUN_ID), "limit": 201},
        )
        response = client.get(
            "/api/benchmark/results",
            params={
                "run_id": str(RUN_ID),
                "outcome": "FN",
                "predicted_personal": "false",
                "need_review": "true",
                "limit": 20,
            },
        )

    assert invalid_outcome.status_code == 422
    assert invalid_limit.status_code == 422
    assert response.status_code == 200
    assert response.json()[0]["outcome"] == "FN"
    assert repository.filters["predicted_personal"] is False
    assert repository.filters["need_review"] is True


def test_benchmark_results_accept_unlabeled_outcome():
    client, repository = make_client()
    with client:
        response = client.get(
            "/api/benchmark/results",
            params={"run_id": str(RUN_ID), "outcome": "UNLABELED"},
        )

    assert response.status_code == 200
    assert repository.filters["outcome"] == "UNLABELED"


def test_benchmark_repository_dependency_reports_missing_target_url(monkeypatch):
    from app.api import benchmark

    monkeypatch.setattr(
        benchmark,
        "get_settings",
        lambda: SimpleNamespace(target_database_url=None),
    )
    application = create_app(lifespan=empty_lifespan)
    with TestClient(application) as client:
        response = client.get(f"/api/benchmark/runs/{RUN_ID}")

    assert response.status_code == 503
    assert response.json()["detail"] == "TARGET_DATABASE_URL is not configured"
