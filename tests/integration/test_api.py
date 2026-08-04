from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from app.api.classification import get_classification_service
from app.main import create_app
from app.schemas.classification import ClassificationResult


class FakeClassificationService:
    def classify_field(self, field):
        return ClassificationResult(
            field_name=field.field_name,
            is_personal=True,
            category="敏感个人信息",
            subcategory="身份标识",
            level="L4",
            confidence=0.9,
            reason="测试判定",
            evidence=[],
            need_review=False,
            decision_path="rag_llm",
        )


@asynccontextmanager
async def empty_lifespan(app):
    yield


def make_client():
    application = create_app(lifespan=empty_lifespan)
    application.dependency_overrides[get_classification_service] = (
        lambda: FakeClassificationService()
    )
    return TestClient(application)


def test_health_endpoint():
    with make_client() as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classification_endpoint_accepts_field_profile():
    with make_client() as client:
        response = client.post(
            "/api/classify",
            json={"field_name": "id_card", "field_cn": "身份证号"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["field_name"] == "id_card"
    assert response.json()["data"]["level"] == "L4"


def test_classification_endpoint_does_not_expose_internal_errors():
    class FailingService:
        def classify_field(self, field):
            raise RuntimeError("secret upstream diagnostic")

    application = create_app(lifespan=empty_lifespan)
    application.dependency_overrides[get_classification_service] = lambda: FailingService()
    with TestClient(application) as client:
        response = client.post("/api/classify", json={"field_name": "id_card"})

    assert response.status_code == 200
    assert response.json()["data"]["reason"] == "系统处理失败，请进行人工复核。"
    assert "secret upstream diagnostic" not in response.text
