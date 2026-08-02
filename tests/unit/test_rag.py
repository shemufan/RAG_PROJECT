from app.rag.chunker import split_knowledge_text
from app.rag.prompt import build_classification_prompt
from app.schemas.classification import Evidence
from app.schemas.field import FieldProfile


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

    assert len(chunks) == 2
    assert chunks[0].metadata["chapter"] == "第一章 总则"
    assert chunks[1].metadata["article"].startswith("第二条")
    assert chunks[1].metadata["sensitivity_level"] == "L4"


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
