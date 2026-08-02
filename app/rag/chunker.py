"""Structure-aware splitting for laws and classification rules."""

import re
import uuid

from langchain_core.documents import Document

_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百0-9]+章")
_ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百0-9]+条")
_RULE_PATTERN = re.compile(r"^#{2,}\s*规则\s*([0-9一二三四五六七八九十]+)[：:]?")


def split_knowledge_text(
    text: str,
    document_name: str,
    *,
    source_type: str,
    version: str,
) -> list[Document]:
    """Split a source at legal article or Markdown rule boundaries."""
    chunks: list[Document] = []
    buffer: list[str] = []
    chapter = "未标注章节"
    article = "全文"

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if not content:
            return
        metadata = {
            "chunk_id": str(uuid.uuid4()),
            "document_name": document_name,
            "source_type": source_type,
            "chapter": chapter,
            "article": article,
            "version": version,
        }
        for level in ("L4", "L3", "L2", "L1"):
            if level in content:
                metadata["sensitivity_level"] = level
                break
        chunks.append(Document(page_content=content, metadata=metadata))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _CHAPTER_PATTERN.match(line):
            flush()
            buffer = []
            chapter = line
            article = "章说明"
            continue
        if _ARTICLE_PATTERN.match(line):
            flush()
            buffer = []
            article = line[:80]
        else:
            rule_match = _RULE_PATTERN.match(line)
            if rule_match:
                if any(not item.startswith("# ") for item in buffer):
                    flush()
                buffer = []
                article = f"规则 {rule_match.group(1)}：{line.split('：', 1)[-1]}"
        buffer.append(line)

    flush()
    return chunks
