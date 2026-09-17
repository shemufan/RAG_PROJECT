from app.rag.chunker import split_knowledge_text
from app.rag.prompt import build_classification_prompt
from app.schemas.classification import Evidence
from app.schemas.field import FieldProfile
from app.schemas.knowledge_quality import ExtractedPage


def test_prompt_contains_validated_field_and_evidence():
    messages = build_classification_prompt(
        FieldProfile(field_name="id_card", field_cn="身份证号"),
        [
            Evidence(
                content="身份证件号码属于敏感个人信息。",
                source="个人信息保护法.txt",
                article="第二十八条",
                score=0.9,
            )
        ],
    )

    assert [message.type for message in messages] == ["system", "human"]
    assert "is_personal" in messages[0].content
    assert "expected_personal" not in messages[0].content
    assert "expected_personal" not in messages[1].content
    assert '"field_name": "id_card"' in messages[1].content
    assert "身份证件号码属于敏感个人信息" in messages[1].content
    assert "仅返回符合结构定义的结果" in messages[0].content
    assert "不可信数据" in messages[0].content


def test_chunker_preserves_chapter_and_article_metadata():
    chunks = split_knowledge_text(
        "第一章 总则\n第一条 公开信息为 L1。\n第二条 身份证号为 L4。",
        "企业规则.txt",
        source_type="legal_document",
        version="v1",
    )

    assert len(chunks) == 3
    assert chunks[0].metadata["chapter"] == "第一章 总则"
    assert chunks[0].page_content == "第一章 总则"
    assert chunks[2].metadata["article"].startswith("第二条")
    assert chunks[2].metadata["sensitivity_level"] == "L4"


def test_chunker_splits_markdown_classification_rules():
    chunks = split_knowledge_text(
        "# 分类规则\n### 规则 1：身份证号\n- 等级: L4\n"
        "### 规则 2：商品名称\n- 等级: L1",
        "classification_rules.md",
        source_type="classification_rule",
        version="v1",
    )

    assert len(chunks) == 2
    assert "身份证号" in chunks[0].page_content
    assert chunks[1].metadata["article"].startswith("规则 2")


def test_chunker_splits_numbered_standard_clauses_and_appendices():
    chunks = split_knowledge_text(
        "1 范围\n范围正文\n3.1 术语\n术语正文\n5.2.3 要求\n"
        "表 1 数据类型\n注：这是说明\n示例：手机号\n附录 A\nA.1 附录要求\n附录正文",
        "标准.pdf",
        source_type="legal_document",
        source_format="pdf",
        version="v2",
        source_sha256="a" * 64,
    )

    assert [chunk.metadata["article"] for chunk in chunks] == [
        "1 范围",
        "3.1 术语",
        "5.2.3 要求",
        "附录说明",
        "A.1 附录要求",
    ]
    assert "表 1 数据类型" in chunks[2].page_content
    assert "示例：手机号" in chunks[2].page_content
    assert chunks[3].metadata["chapter"] == "附录 A"
    assert all(chunk.metadata["source_sha256"] == "a" * 64 for chunk in chunks)


def test_chunker_preserves_page_range_for_standard_clause():
    pages = [
        ExtractedPage(page_number=3, text="1 范围\n第一页正文", extraction_method="native"),
        ExtractedPage(page_number=4, text="续页正文\n2 要求\n第二条正文", extraction_method="ocr"),
    ]

    from app.rag.chunker import split_knowledge_pages

    chunks = split_knowledge_pages(
        pages,
        "标准.pdf",
        source_type="legal_document",
        source_format="pdf",
        version="v2",
        source_sha256="b" * 64,
    )

    assert chunks[0].metadata["page_start"] == 3
    assert chunks[0].metadata["page_end"] == 4
    assert chunks[1].metadata["page_start"] == 4
    assert chunks[1].metadata["page_end"] == 4


def test_chunker_splits_long_clause_with_overlap_and_stable_ids():
    text = "1 范围\n" + "甲" * 2600
    kwargs = {
        "source_type": "legal_document",
        "source_format": "pdf",
        "version": "v2",
        "source_sha256": "c" * 64,
    }

    first = split_knowledge_text(text, "标准.pdf", **kwargs)
    second = split_knowledge_text(text, "标准.pdf", **kwargs)

    assert len(first) >= 2
    assert all(len(chunk.page_content) <= 2000 for chunk in first)
    assert first[0].page_content[-150:] == first[1].page_content[:150]
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
