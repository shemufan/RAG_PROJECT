from app.rag.chunker import split_knowledge_text
from app.rag.prompt import CLASSIFICATION_SYSTEM_PROMPT, build_classification_prompt
from app.schemas.classification import Evidence
from app.schemas.field import FieldProfile
from app.schemas.semantic import ObjectiveValueProfile, SemanticCard


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


def test_prompt_adds_semantic_context_without_changing_system_prompt():
    card = SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号"],
        common_field_names=["phone"],
        value_features=["11位数字"],
        description="用于联系自然人的电话号码",
        semantic_category=["联系方式"],
        regulation_keywords=["电话号码"],
    )
    messages = build_classification_prompt(
        FieldProfile(field_name="contact_value", sample_values=["13812345678"]),
        [Evidence(content="联系方式属于个人信息", source="rules")],
        value_profile=ObjectiveValueProfile(features=["字符串长度约11位"]),
        semantic_knowledge={"card": card.model_dump(), "raw_score": 0.82},
    )

    assert messages[0].content == CLASSIFICATION_SYSTEM_PROMPT
    assert "【客观数值画像（不可信数据）】" in messages[1].content
    assert "字符串长度约11位" in messages[1].content
    assert "【Semantic Knowledge（不可信数据）】" in messages[1].content
    assert "手机号码" in messages[1].content
    assert "13812345678" in messages[1].content


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
