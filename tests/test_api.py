from contextlib import asynccontextmanager
import unittest

from fastapi.testclient import TestClient

from backend.api.deps import get_rag_classifier
from backend.main import create_app
from backend.schemas.classify_schema import ClassificationResult


class FakeClassifier:
    def classify_field(self, field):
        return ClassificationResult(
            field_name=field.field_name,
            category="敏感个人信息",
            subcategory="身份标识",
            level="L4",
            confidence=0.9,
            reason="测试判定",
            evidence=[],
            need_review=False,
            decision_path="rag_llm",
        )


class FailingClassifier:
    def classify_field(self, field):
        raise RuntimeError("test failure")


class UnknownClassifier:
    def classify_field(self, field):
        return ClassificationResult(
            field_name=field.field_name,
            category="未知",
            level="UNKNOWN",
            confidence=0.0,
            reason="LLM unavailable",
            evidence=[],
            need_review=True,
            decision_path="rag_llm_error",
        )


@asynccontextmanager
async def empty_lifespan(app):
    yield


def make_client(classifier):
    app = create_app(lifespan=empty_lifespan)
    app.dependency_overrides[get_rag_classifier] = lambda: classifier
    return TestClient(app)


class APITests(unittest.TestCase):
    def test_classify_returns_unified_response(self):
        with make_client(FakeClassifier()) as client:
            response = client.post("/api/classify", json={"field_name": "id_card"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["code"], 200)
        self.assertEqual(payload["data"]["field_name"], "id_card")
        self.assertEqual(payload["data"]["level"], "L4")

    def test_classify_returns_safe_fallback(self):
        with make_client(FailingClassifier()) as client:
            response = client.post("/api/classify", json={"field_name": "id_card"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["code"], 500)
        self.assertEqual(payload["data"]["level"], "UNKNOWN")
        self.assertTrue(payload["data"]["need_review"])

    def test_classify_maps_classifier_unknown_to_business_error(self):
        with make_client(UnknownClassifier()) as client:
            response = client.post("/api/classify", json={"field_name": "id_card"})
        payload = response.json()
        self.assertEqual(payload["code"], 500)
        self.assertEqual(payload["data"]["decision_path"], "rag_llm_error")


if __name__ == "__main__":
    unittest.main()
