"""A-database persistence and row reads for benchmark inputs."""

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.benchmark import BenchmarkImportRow


class BenchmarkSourceError(RuntimeError):
    """Raised when benchmark input persistence fails."""


class BenchmarkSourceRepository:
    """Import labeled cases into A without exposing labels to the classifier."""

    def __init__(self, database_url: str | None = None, *, engine=None):
        if engine is None:
            if not database_url:
                raise ValueError("SOURCE_DATABASE_URL is not configured")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine

    def import_batch(
        self,
        batch_name: str,
        rows: list[BenchmarkImportRow],
    ) -> int:
        """Insert a fully parsed batch in one transaction and keep duplicates by row."""
        statement = text(
            """
            INSERT INTO benchmark_field_input (
                batch_name, source_dataset, source_row_number, field_name,
                sample_values_json, expected_personal, created_at
            ) VALUES (
                :batch_name, :source_dataset, :source_row_number, :field_name,
                :sample_values_json, :expected_personal, :created_at
            )
            ON DUPLICATE KEY UPDATE benchmark_id = benchmark_id
            """
        )
        now = datetime.now(timezone.utc)
        parameters = [
            {
                "batch_name": batch_name,
                "source_dataset": row.source_dataset,
                "source_row_number": row.source_row_number,
                "field_name": row.field_name,
                "sample_values_json": json.dumps(
                    row.sample_values,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "expected_personal": row.expected_personal,
                "created_at": now,
            }
            for row in rows
        ]
        if not parameters:
            return 0
        try:
            with self.engine.begin() as connection:
                connection.execute(statement, parameters)
        except SQLAlchemyError as exc:
            raise BenchmarkSourceError("failed to import benchmark batch") from exc
        return len(parameters)
