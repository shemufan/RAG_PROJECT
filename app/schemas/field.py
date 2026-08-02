"""Input model for a single enterprise data field."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

FieldName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=256)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)]
SampleValue = Annotated[str, StringConstraints(strip_whitespace=True, max_length=256)]


class FieldProfile(BaseModel):
    """Validated metadata used to classify one field."""

    model_config = ConfigDict(extra="forbid")

    field_name: FieldName
    field_cn: ShortText | None = None
    field_comment: LongText | None = None
    data_type: ShortText | None = None
    sample_values: list[SampleValue] = Field(default_factory=list, max_length=5)
    business_domain: ShortText = "general"
    table_name: ShortText | None = None
    database_name: ShortText | None = None
    source_system: ShortText | None = None
