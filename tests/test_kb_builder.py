import tempfile
import unittest
from pathlib import Path

from backend.scripts.rebuild_kb import load_source_documents
from backend.utils.chunker import split_by_article


class KnowledgeBuilderTests(unittest.TestCase):
    def test_chunker_preserves_chinese_article_metadata(self):
        chunks = split_by_article(
            "第一章 总则\n第一条 公开信息为L1。\n第二条 身份证号为L4。",
            "企业规则.txt",
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["chapter"], "第一章 总则")
        self.assertEqual(chunks[1].metadata["article"], "第二条 身份证号为L4。")

    def test_chunker_splits_markdown_rules_individually(self):
        chunks = split_by_article(
            "# 规则\n### 规则 1：身份证号\n- 等级: L4\n### 规则 2：商品名称\n- 等级: L1",
            "rules.md",
        )
        self.assertEqual(len(chunks), 2)
        self.assertIn("身份证号", chunks[0].page_content)
        self.assertIn("规则 2", chunks[1].metadata["article"])

    def test_loader_reads_rules_and_txt_files_as_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "knowledge_docs"
            docs_dir.mkdir()
            (root / "rules.md").write_text("# 规则\n身份证号为 L4", encoding="utf-8")
            (docs_dir / "法律.txt").write_text("第一条 个人信息受保护", encoding="utf-8")
            chunks = load_source_documents(root)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("身份证号" in chunk.page_content for chunk in chunks))
        self.assertTrue(all("document_name" in chunk.metadata for chunk in chunks))

    def test_loader_decodes_legacy_gb18030_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "knowledge_docs"
            docs_dir.mkdir()
            (docs_dir / "规范.txt").write_bytes("第一条 个人信息安全规范".encode("gb18030"))
            chunks = load_source_documents(root)

        self.assertIn("个人信息安全规范", chunks[0].page_content)


if __name__ == "__main__":
    unittest.main()
