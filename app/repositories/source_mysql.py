"""Read-only mapping from MySQL physical fields to validated field profiles."""

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.field import FieldProfile

logger = logging.getLogger(__name__)

BUSINESS_DOMAINS = {
    "employee": "hr",
    "customer_account": "customer",
    "customer_order": "commerce",
    "product": "product",
}

METADATA_SQL = """
SELECT
    c.TABLE_SCHEMA,
    c.TABLE_NAME,
    t.TABLE_COMMENT,
    c.COLUMN_NAME,
    c.COLUMN_TYPE,
    c.COLUMN_COMMENT,
    c.IS_NULLABLE,
    c.COLUMN_KEY,
    c.ORDINAL_POSITION
FROM information_schema.COLUMNS AS c
JOIN information_schema.TABLES AS t
  ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
 AND t.TABLE_NAME = c.TABLE_NAME
WHERE c.TABLE_SCHEMA = :database_name
  AND t.TABLE_TYPE = 'BASE TABLE'
{table_filter}
ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
"""


def validate_sample_limit(sample_limit: int) -> int:
    """Enforce the repository sampling boundary."""
    if not 0 <= sample_limit <= 5:
        raise ValueError("sample_limit 必须在 0 到 5 之间")
    return sample_limit


def business_domain_for_table(table_name: str) -> str:
    """Map known demo tables to stable business domains."""
    return BUSINESS_DOMAINS.get(table_name, "general")


def _mask_middle(value: str, prefix: int, suffix: int) -> str:
    if len(value) <= prefix + suffix:
        return value[:50]
    return f"{value[:prefix]}{'*' * (len(value) - prefix - suffix)}{value[-suffix:]}"


def mask_sample_value(field_name: str, value: Any) -> str:
    """Mask one sample according to the physical field name."""
    rendered = str(value)
    normalized_name = field_name.lower()
    if "id_card" in normalized_name or "identity" in normalized_name:
        return _mask_middle(rendered, 4, 4)[:50]
    if "phone" in normalized_name or "mobile" in normalized_name:
        return _mask_middle(rendered, 3, 4)[:50]
    if "bank_card" in normalized_name or "card_no" in normalized_name:
        return _mask_middle(rendered, 4, 4)[:50]
    if "email" in normalized_name and "@" in rendered:
        local, domain = rendered.split("@", 1)
        masked_local = f"{local[:1]}****" if local else "****"
        return f"{masked_local}@{domain}"[:50]
    return rendered[:50]


def map_metadata_row(row: Mapping[str, Any], samples: list[Any]) -> FieldProfile:
    """Convert one information-schema row and raw samples to a field profile."""
    masked_samples = []
    for value in samples:
        if value is None:
            continue
        masked = mask_sample_value(row["COLUMN_NAME"], value)
        if masked not in masked_samples:
            masked_samples.append(masked)
        if len(masked_samples) == 5:
            break
    column_comment = row.get("COLUMN_COMMENT") or None
    return FieldProfile(
        source_system="mysql",
        database_name=row["TABLE_SCHEMA"],
        table_name=row["TABLE_NAME"],
        table_comment=row.get("TABLE_COMMENT") or None,
        field_name=row["COLUMN_NAME"],
        field_cn=column_comment,
        field_comment=column_comment,
        data_type=row["COLUMN_TYPE"],
        is_nullable=str(row["IS_NULLABLE"]).upper() == "YES",
        column_key=row.get("COLUMN_KEY") or None,
        sample_values=masked_samples,
        business_domain=business_domain_for_table(row["TABLE_NAME"]),
    )


class SourceMySQLRepository:
    """Read MySQL metadata and masked field samples without modifying source data."""

    def __init__(self, database_url: str | None = None, *, engine=None):
        if engine is None:
            if not database_url:
                raise ValueError("SOURCE_DATABASE_URL 未配置")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine
        self.database_name = engine.url.database
        if not self.database_name:
            raise ValueError("SOURCE_DATABASE_URL 必须包含数据库名")

    def scan_fields(
        self,
        sample_limit: int = 3,
        table_names: list[str] | None = None,
    ) -> list[FieldProfile]:
        """Return physical source fields in deterministic table and column order."""
        limit = validate_sample_limit(sample_limit)
        metadata = self._load_metadata(table_names)
        profiles = []
        for row in metadata:
            profiles.append(map_metadata_row(row, self._load_samples(row, limit)))
        return profiles

    def _load_metadata(self, table_names: list[str] | None) -> list[dict[str, Any]]:
        if table_names == []:
            return []
        table_filter = ""
        parameters: dict[str, Any] = {"database_name": self.database_name}
        statement = text(METADATA_SQL.format(table_filter=table_filter))
        if table_names is not None:
            table_filter = "  AND c.TABLE_NAME IN :table_names"
            parameters["table_names"] = table_names
            statement = text(METADATA_SQL.format(table_filter=table_filter)).bindparams(
                bindparam("table_names", expanding=True)
            )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement, parameters).mappings()]

    def _load_samples(self, row: Mapping[str, Any], sample_limit: int) -> list[Any]:
        if sample_limit == 0:
            return []
        quote = self.engine.dialect.identifier_preparer.quote
        schema = quote(row["TABLE_SCHEMA"])
        table = quote(row["TABLE_NAME"])
        column = quote(row["COLUMN_NAME"])
        statement = text(
            f"SELECT DISTINCT {column} AS sample_value "
            f"FROM {schema}.{table} "
            f"WHERE {column} IS NOT NULL LIMIT :sample_limit"
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(statement, {"sample_limit": sample_limit}).mappings()
                return [item["sample_value"] for item in rows]
        except SQLAlchemyError:
            logger.warning(
                "字段取样失败：%s.%s.%s",
                row["TABLE_SCHEMA"],
                row["TABLE_NAME"],
                row["COLUMN_NAME"],
                exc_info=True,
            )
            return []
