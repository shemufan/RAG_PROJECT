"""Deterministic catalog-versus-tabular CSV mode selection."""

import re

from app.schemas.csv_input import InputMode, ResolvedInputMode
from app.services.csv_reader import CSVInputError

FIELD_NAME_ALIASES = {"字段名", "field_name"}
CATALOG_METADATA_ALIASES = {
    "字段中文名",
    "field_cn",
    "字段说明",
    "field_comment",
    "数据类型",
    "data_type",
    "业务域",
    "business_domain",
    "表名",
    "table_name",
    "数据库名",
    "database_name",
    "来源系统",
    "source_system",
}
SAMPLE_HEADER = re.compile(r"^(?:样本(\d+)|sample_?(\d+))$", re.IGNORECASE)


class CSVModeError(CSVInputError):
    """Raised when automatic mode selection would be unsafe."""


def sample_number(header: str) -> int | None:
    match = SAMPLE_HEADER.fullmatch(header)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def resolve_csv_mode(headers: list[str], requested_mode: InputMode) -> ResolvedInputMode:
    if requested_mode != "auto":
        return requested_mode
    has_field_name = any(header in FIELD_NAME_ALIASES for header in headers)
    has_samples = any(sample_number(header) is not None for header in headers)
    if not (has_field_name and has_samples):
        return "tabular"
    recognized = FIELD_NAME_ALIASES | CATALOG_METADATA_ALIASES
    unknown = [
        header
        for header in headers
        if header not in recognized and sample_number(header) is None
    ]
    if unknown:
        raise CSVModeError("ambiguous CSV input mode; specify catalog or tabular")
    return "catalog"
