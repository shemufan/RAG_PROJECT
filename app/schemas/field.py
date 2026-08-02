"""Input model for a single enterprise data field."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

FieldName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
RequiredName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=256)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]
SampleValue = Annotated[str, StringConstraints(strip_whitespace=True, max_length=50)]


class FieldProfile(BaseModel):
    """Validated metadata used to classify one field."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_system: RequiredName = "manual"
    database_name: RequiredName = "manual"
    table_name: RequiredName = "manual"
    table_comment: LongText | None = None
    field_name: FieldName
    field_cn: ShortText | None = None
    field_comment: LongText | None = None
    data_type: RequiredName = "unknown"
    is_nullable: bool = True
    column_key: ShortText | None = None
    sample_values: list[SampleValue] = Field(default_factory=list, max_length=5)
    business_domain: ShortText = "general"
