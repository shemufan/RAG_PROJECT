"""Convert ordinary column-oriented business CSV files into canonical cases."""

from pathlib import Path

from app.schemas.csv_input import CSVFieldCase, CSVInputBatch
from app.schemas.field import FieldProfile
from app.services.csv_reader import CSVReader


class TabularCSVAdapter:
    def __init__(self, reader: CSVReader | None = None):
        self.reader = reader or CSVReader()

    def load(self, path: str | Path) -> CSVInputBatch:
        inspection = self.reader.inspect(path)
        samples: dict[str, list[str]] = {header: [] for header in inspection.headers}
        for _, row in self.reader.iter_rows(path, inspection):
            for header in inspection.headers:
                value = row[header].strip()
                if value and len(samples[header]) < 5:
                    samples[header].append(value[:50])
        cases = [
            CSVFieldCase(
                case_index=index,
                field_profile=FieldProfile(
                    source_system="csv",
                    database_name="csv_source",
                    table_name="tabular_input",
                    field_name=header,
                    data_type="unknown",
                    business_domain="general",
                    sample_values=samples[header],
                ),
            )
            for index, header in enumerate(inspection.headers, start=1)
        ]
        return CSVInputBatch(
            source_name=inspection.source_name,
            source_fingerprint=inspection.source_fingerprint,
            input_mode="tabular",
            cases=cases,
        )
