"""Build the mainline retrieval query from field names and sample values."""

from app.schemas.field import FieldProfile


def build_retrieval_query(field: FieldProfile) -> str:
    values = {
        "field_name": field.field_name,
        "sample_values": "、".join(field.sample_values),
    }
    return "\n".join(f"{key}: {value}" for key, value in values.items() if value)
