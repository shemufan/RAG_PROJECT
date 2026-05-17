# knowledge/chunker.py

import re
import uuid
from langchain_core.documents import Document


def split_by_article(
    text: str, document_name: str, source_type: str = "enterprise_rule", domain: str = "general"
):
    """
    简单版结构化分块：
    按“第X章 / 第X条”进行切分。
    Layer1 阶段先做到可用，不追求完美。
    """
    chunks = []

    current_chapter = "未知章节"

    # 按行处理
    lines = text.splitlines()
    buffer = []
    current_article = "未知条款"

    def flush():
        if not buffer:
            return

        content = "\n".join(buffer).strip()
        if not content:
            return

        metadata = {
            "chunk_id": str(uuid.uuid4()),
            "document_name": document_name,
            "source_type": source_type,
            "domain": domain,
            "hierarchy_level": f"{current_chapter}-{current_article}",
            "chapter": current_chapter,
            "article": current_article,
            "version": "2026",
        }

        # 简单识别 level
        for level in ["L4", "L3", "L2", "L1"]:
            if level in content:
                metadata["sensitivity_level"] = level
                break

        chunks.append(Document(page_content=content, metadata=metadata))

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 正则表达式
        if re.match(r"^第[一二三四五六七八九十0-9]+章", line):
            flush()
            buffer = []
            current_chapter = line
            current_article = "章说明"
            continue

        if re.match(r"^第[一二三四五六七八九十0-9]+条", line):
            flush()
            buffer = []
            current_article = line[:20]

        buffer.append(line)

    flush()
    return chunks
