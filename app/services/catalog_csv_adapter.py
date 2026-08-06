"""Convert row-oriented field catalog CSV files into canonical cases."""

from pathlib import Path

from app.schemas.csv_input import CSVFieldCase, CSVInputBatch
from app.schemas.field import FieldProfile
from app.services.csv_mode_detector import FIELD_NAME_ALIASES, sample_number
from app.services.csv_reader import CSVInputError, CSVReader

ALIASES = {
    "field_cn": ("字段中文名", "field_cn"),
    "field_comment": ("字段说明", "field_comment"),
    "data_type": ("数据类型", "data_type"),
    "business_domain": ("业务域", "business_domain"),
    "table_name": ("表名", "table_name"),
    "database_name": ("数据库名", "database_name"),
    "source_system": ("来源系统", "source_system"),
}


class CatalogCSVAdapter:
    def __init__(self, reader: CSVReader | None = None):
        self.reader = reader or CSVReader()

    def load(
        self,
        path: str | Path,
        *,
        field_name_column: str | None = None,
        sample_columns: list[str] | None = None,
    ) -> CSVInputBatch:
        inspection = self.reader.inspect(path)
        name_column = field_name_column or next(
            (header for header in inspection.headers if header in FIELD_NAME_ALIASES),
            None,
        )
        if not name_column or name_column not in inspection.headers:
            raise CSVInputError("catalog field-name column is missing")
        if sample_columns is None:
            numbered = [
                (number, header)
                for header in inspection.headers
                if (number := sample_number(header)) is not None
            ]
            sample_columns = [header for _, header in sorted(numbered)]
        missing = [column for column in sample_columns if column not in inspection.headers]
        if missing:
            raise CSVInputError("configured sample column is missing")

        cases = []
        for case_index, (row_number, row) in enumerate(
            self.reader.iter_rows(path, inspection),
            start=1,
        ):
            field_name = row[name_column].strip()
            if not field_name:
                raise CSVInputError(f"row {row_number} has an empty field name")
            samples = [row[column].strip()[:50] for column in sample_columns]
            samples = [sample for sample in samples if sample][:5]
            values = self._metadata(row)
            cases.append(
                CSVFieldCase(
                    case_index=case_index,
                    field_profile=FieldProfile(
                        field_name=field_name,
                        sample_values=samples,
                        source_system=values.get("source_system") or "csv",
                        database_name=values.get("database_name") or "csv_source",
                        table_name=values.get("table_name") or "catalog_input",
                        data_type=values.get("data_type") or "unknown",
                        business_domain=values.get("business_domain") or "general",
                        field_cn=values.get("field_cn"),
                        field_comment=values.get("field_comment"),
                    ),
                )
            )
        return CSVInputBatch(
            source_name=inspection.source_name,
            source_fingerprint=inspection.source_fingerprint,
            input_mode="catalog",
            cases=cases,
        )

    @staticmethod
    def _metadata(row: dict[str, str]) -> dict[str, str | None]:
        values = {}
        for target, aliases in ALIASES.items():
            raw = next((row[name] for name in aliases if name in row), "")
            values[target] = raw.strip() or None
        return values
