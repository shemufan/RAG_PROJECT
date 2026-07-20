import unittest

from langchain_core.documents import Document

from backend.schemas.classify_schema import ClassificationOutput, FieldProfile
from backend.services.rag_classifier import RAGClassifier


class FakeStore:
    def __init__(self):
        self.query = None

    def similarity_search_with_relevance_scores(self, query, k=3):
        self.query = query
        return [
            (
                Document(
                    page_content="身份证件号码属于敏感个人信息。",
                    metadata={
                        "document_name": "个人信息保护法.txt",
                        "article": "第二十八条",
                    },
                ),
                0.91,
            )
        ]


class FakeLLM:
    def classify(self, user_message):
        assert "身份证件号码属于敏感个人信息" in user_message
        return ClassificationOutput(
            category="敏感个人信息",
            subcategory="身份标识",
            level="L4",
            confidence=0.92,
            reason="依据《个人信息保护法》第二十八条判定。",
            need_review=False,
        )


class RAGClassifierTests(unittest.TestCase):
    def test_builds_query_and_returns_evidence(self):
        store = FakeStore()
        classifier = RAGClassifier(store, FakeLLM())
        result = classifier.classify_field(
            FieldProfile(
                field_name="id_card",
                field_cn="身份证号",
                field_comment="用户身份证号码",
                sample_values=["340************1234"],
            )
        )
        self.assertIn("id_card", store.query)
        self.assertIn("身份证号", store.query)
        self.assertEqual(result.level, "L4")
        self.assertEqual(result.category, "敏感个人信息")
        self.assertEqual(result.evidence[0].article, "第二十八条")
        self.assertEqual(result.decision_path, "rag_llm")

    def test_returns_review_result_when_llm_fails(self):
        class FailingLLM:
            def classify(self, user_message):
                raise RuntimeError("provider unavailable")

        classifier = RAGClassifier(FakeStore(), FailingLLM())
        result = classifier.classify_field(FieldProfile(field_name="unknown_field"))
        self.assertEqual(result.level, "UNKNOWN")
        self.assertTrue(result.need_review)
        self.assertEqual(result.decision_path, "rag_llm_error")
        self.assertIn("provider unavailable", result.reason)


if __name__ == "__main__":
    unittest.main()
import unittest
