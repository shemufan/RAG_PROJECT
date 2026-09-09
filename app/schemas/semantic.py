"""Validated models for the Experiment E Semantic Knowledge Base."""

from pydantic import BaseModel, Field, field_validator


class SemanticCard(BaseModel):
    """One general domain-semantic description of a field type."""

    semantic_type: str
    aliases: list[str] = Field(min_length=1)
    common_field_names: list[str] = Field(min_length=1)
    value_features: list[str] = Field(min_length=1)
    description: str
    semantic_category: list[str] = Field(min_length=1)
    regulation_keywords: list[str] = Field(min_length=1)

    @field_validator("semantic_type", "description")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("semantic card text must not be empty")
        return cleaned

    @field_validator(
        "aliases",
        "common_field_names",
        "value_features",
        "semantic_category",
        "regulation_keywords",
    )
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("semantic card lists must not contain empty values")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("semantic card lists must not contain duplicates")
        return cleaned


class SemanticRetrievalResult(BaseModel):
    """One retrieved Semantic Card with its unmodified store score."""

    card: SemanticCard
    raw_score: float


class ObjectiveValueProfile(BaseModel):
    """Only observable facts; deliberately has no semantic candidate field."""

    features: list[str] = Field(default_factory=list)
