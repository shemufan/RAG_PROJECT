from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.api.pipeline import get_database_pipeline, get_target_repository
from app.main import create_app
from app.schemas.pipeline import (
    ClassificationEvidenceRow,
    PipelineSummary,
    RunDetail,
)

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 2, tzinfo=timezone.utc)


class FakePipeline:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return PipelineSummary(
            run_id=RUN_ID,
            source_database="enterprise_source",
            total_fields=4,
            success_fields=4,
            review_fields=1,
            failed_fields=0,
            status="SUCCESS",
            started_at=NOW,
            finished_at=NOW,
        )


class FakeTargetRepository:
    def __init__(self):
        self.result_filters = None

    def get_run(self, run_id):
        if run_id != RUN_ID:
            return None
        return RunDetail(
            run_id=RUN_ID,
            source_system="mysql",
            source_database="enterprise_source",
            status="SUCCESS",
            total_fields=4,
            success_fields=4,
            review_fields=1,
            failed_fields=0,
            model_name="deepseek-chat",
            knowledge_base_version="v1",
            started_at=NOW,
            finished_at=NOW,
        )

    def query_results(self, **filters):
        self.result_filters = filters
        return []

    def get_result_evidence(self, result_id):
        return [
            ClassificationEvidenceRow(
                evidence_id=1,
                result_id=result_id,
                rank_no=1,
                document_name="个人信息保护法.txt",
                article="第二十八条",
                content="敏感个人信息定义",
                relevance_score=0.9,
                chunk_id="chunk-28",
            )
        ]


@asynccontextmanager
async def empty_lifespan(app):
    yield


def make_client():
    pipeline = FakePipeline()
    target = FakeTargetRepository()
    application = create_app(lifespan=empty_lifespan)
    application.dependency_overrides[get_database_pipeline] = lambda: pipeline
    application.dependency_overrides[get_target_repository] = lambda: target
    return TestClient(application), pipeline, target, application


def test_pipeline_run_endpoint_returns_summary():
    client, pipeline, _, _ = make_client()

    with client:
        response = client.post(
            "/api/pipeline/run",
            json={"sample_limit": 3, "table_names": None, "continue_on_error": True},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
    assert pipeline.requests[0].sample_limit == 3


def test_run_endpoint_returns_status_and_404_for_missing_run():
    client, _, _, _ = make_client()

    with client:
        found = client.get(f"/api/runs/{RUN_ID}")
        missing = client.get(f"/api/runs/{uuid4()}")

    assert found.status_code == 200
    assert found.json()["source_database"] == "enterprise_source"
    assert missing.status_code == 404


def test_results_query_validates_and_forwards_filters():
    client, _, target, _ = make_client()

    with client:
        invalid = client.get("/api/results?limit=201")
        valid = client.get(
            "/api/results?database_name=enterprise_source&table_name=employee"
            "&level=L4&need_review=true&limit=20&offset=5"
        )

    assert invalid.status_code == 422
    assert valid.status_code == 200
    assert target.result_filters["database_name"] == "enterprise_source"
    assert target.result_filters["table_name"] == "employee"
    assert target.result_filters["level"] == "L4"
    assert target.result_filters["need_review"] is True
    assert target.result_filters["limit"] == 20
    assert target.result_filters["offset"] == 5


def test_result_evidence_endpoint_returns_ranked_rows():
    client, _, _, _ = make_client()

    with client:
        response = client.get("/api/results/41/evidence")

    assert response.status_code == 200
    assert response.json()[0]["rank_no"] == 1
    assert response.json()[0]["document_name"] == "个人信息保护法.txt"


def test_openapi_registers_pipeline_and_existing_classification_paths():
    _, _, _, application = make_client()
    paths = application.openapi()["paths"]

    assert "/api/pipeline/run" in paths
    assert "/api/runs/{run_id}" in paths
    assert "/api/results" in paths
    assert "/api/results/{result_id}/evidence" in paths
    assert "/api/classify" in paths
