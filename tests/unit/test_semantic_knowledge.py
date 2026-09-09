import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT
from app.schemas.semantic import SemanticCard
from app.services.semantic_knowledge_service import (
    load_semantic_cards,
    semantic_card_to_document,
)


SEMANTIC_FILE = PROJECT_ROOT / "data" / "semantic_knowledge" / "semantic_cards.json"


def _phone_card() -> SemanticCard:
    return SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号", "phone"],
        common_field_names=["phone", "mobile"],
        value_features=["通常由数字组成", "中国大陆手机号通常为11位"],
        description="用于联系自然人的电话号码",
        semantic_category=["联系方式", "个人信息"],
        regulation_keywords=["手机号码", "联系方式"],
    )


def test_semantic_card_rejects_empty_and_duplicate_list_values():
    payload = _phone_card().model_dump()
    payload["aliases"] = ["phone", "phone"]
    with pytest.raises(ValidationError):
        SemanticCard.model_validate(payload)

    payload = _phone_card().model_dump()
    payload["description"] = " "
    with pytest.raises(ValidationError):
        SemanticCard.model_validate(payload)


def test_semantic_card_document_is_deterministic_and_round_trippable():
    document = semantic_card_to_document(_phone_card())

    assert document.page_content == (
        "语义类型：手机号码\n"
        "别名：手机号、phone\n"
        "常见字段名：phone、mobile\n"
        "数据特征：通常由数字组成；中国大陆手机号通常为11位\n"
        "业务含义：用于联系自然人的电话号码\n"
        "语义类别：联系方式、个人信息\n"
        "法规关键词：手机号码、联系方式"
    )
    assert document.metadata["semantic_type"] == "手机号码"
    assert json.loads(document.metadata["aliases_json"]) == ["手机号", "phone"]


def test_loader_rejects_duplicate_semantic_types(tmp_path: Path):
    card = _phone_card().model_dump()
    path = tmp_path / "cards.json"
    path.write_text(json.dumps([card, card], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate semantic_type"):
        load_semantic_cards(path)


def test_tracked_semantic_cards_cover_common_types_without_ground_truth():
    cards = load_semantic_cards(SEMANTIC_FILE)

    assert 20 <= len(cards) <= 50
    semantic_types = {card.semantic_type for card in cards}
    assert {
        "姓名",
        "手机号码",
        "固定电话",
        "身份证号码",
        "邮箱",
        "地址",
        "银行卡号",
        "IP地址",
        "MAC地址",
        "设备ID",
        "用户ID",
        "客户编号",
        "医疗信息",
    } <= semantic_types
    raw = SEMANTIC_FILE.read_text(encoding="utf-8")
    assert "expected_personal" not in raw
    assert "ground_truth" not in raw
    assert "benchmark" not in raw.casefold()
