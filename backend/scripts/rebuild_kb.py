"""Rebuild the dedicated Week3 Chroma collection from UTF-8 source files."""

import logging
from pathlib import Path

from backend.core.config import DATA_DIR
from backend.services.knowledge_base_service import read_source_text
from backend.utils.chunker import split_by_article

logger = logging.getLogger(__name__)


def load_source_documents(data_root: str | Path = DATA_DIR):
    root = Path(data_root)
    paths = [root / "rules.md", *sorted((root / "knowledge_docs").glob("*.txt"))]
    chunks = []
    for path in paths:
        if not path.is_file():
            continue
        text = read_source_text(path)
        source_type = "classification_rule" if path.name == "rules.md" else "legal_document"
        chunks.extend(split_by_article(text, path.name, source_type=source_type))
    if not chunks:
        raise RuntimeError(f"未在 {root} 找到可入库的 UTF-8 知识文档")
    return chunks


def main():
    from backend.storage.chroma_store import ChromaStore
    from backend.services.embedding_service import EmbeddingService

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = ChromaStore(EmbeddingService())
    chunks = load_source_documents()
    store.reset()
    store.add_documents(chunks)
    print(f"知识库重建完成：{store.count()} 个知识块")
    for query in ("身份证号 身份证件号码", "员工工资 薪酬", "商品名称 公开产品信息"):
        matches = store.similarity_search_with_relevance_scores(query, k=1)
        if not matches:
            raise RuntimeError(f"冒烟检索无结果: {query}")
        document, score = matches[0]
        if document.metadata.get("document_name") != "rules.md":
            raise RuntimeError(
                f"冒烟检索未命中分类规则: {query} -> "
                f"{document.metadata.get('document_name')}"
            )
        print(f"[检索] {query} -> {document.metadata.get('document_name')} ({score:.3f})")


if __name__ == "__main__":
    main()
