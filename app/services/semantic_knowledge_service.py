"""Load and vectorize general Semantic Knowledge Base cards."""

import json
from pathlib import Path

from langchain_core.documents import Document

from app.schemas.semantic import SemanticCard


def load_semantic_cards(path: str | Path) -> list[SemanticCard]:
    """Validate every card before any vector collection is mutated."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("semantic knowledge file must contain a JSON array")
    cards = [SemanticCard.model_validate(item) for item in payload]
    semantic_types = [card.semantic_type for card in cards]
    if len(set(semantic_types)) != len(semantic_types):
        raise ValueError("duplicate semantic_type in semantic knowledge file")
    if not cards:
        raise ValueError("semantic knowledge file contains no cards")
    return cards


def semantic_card_to_document(card: SemanticCard) -> Document:
    """Create deterministic embedding text and scalar Chroma metadata."""

    text = "\n".join(
        (
            f"语义类型：{card.semantic_type}",
            f"别名：{'、'.join(card.aliases)}",
            f"常见字段名：{'、'.join(card.common_field_names)}",
            f"数据特征：{'；'.join(card.value_features)}",
            f"业务含义：{card.description}",
            f"语义类别：{'、'.join(card.semantic_category)}",
            f"法规关键词：{'、'.join(card.regulation_keywords)}",
        )
    )
    metadata = {"semantic_type": card.semantic_type}
    for name in (
        "aliases",
        "common_field_names",
        "value_features",
        "semantic_category",
        "regulation_keywords",
    ):
        metadata[f"{name}_json"] = json.dumps(
            getattr(card, name), ensure_ascii=False
        )
    metadata["description"] = card.description
    return Document(page_content=text, metadata=metadata)


def load_semantic_documents(path: str | Path) -> list[Document]:
    return [semantic_card_to_document(card) for card in load_semantic_cards(path)]


class SemanticKnowledgeService:
    """Replace only the independent Semantic vector collection."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def rebuild(self, documents: list[Document]) -> int:
        self.vector_store.reset()
        self.vector_store.add_documents(documents)
        return len(documents)

