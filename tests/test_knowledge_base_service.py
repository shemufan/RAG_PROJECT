import tempfile
import unittest
from pathlib import Path

from backend.services.knowledge_base_service import KnowledgeBaseService


class FakeStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        return list(range(len(documents)))


class KnowledgeBaseServiceTests(unittest.TestCase):
    def test_update_from_paths_reads_and_adds_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "规则.txt"
            path.write_text("第一条 身份证号属于敏感个人信息", encoding="utf-8")
            store = FakeStore()

            added = KnowledgeBaseService(store).update_from_paths([path])

        self.assertEqual(added, 1)
        self.assertEqual(len(store.documents), 1)
        self.assertEqual(store.documents[0].metadata["document_name"], "规则.txt")

    def test_update_from_paths_ignores_empty_input(self):
        store = FakeStore()

        added = KnowledgeBaseService(store).update_from_paths([])

        self.assertEqual(added, 0)
        self.assertEqual(store.documents, [])


if __name__ == "__main__":
    unittest.main()
