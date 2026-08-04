"""A-database persistence and row reads for benchmark inputs."""

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.benchmark import (
    BenchmarkCase,
    BenchmarkImportRow,
    BenchmarkImportSummary,
)
from app.schemas.field import FieldProfile


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
    ) -> BenchmarkImportSummary:
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
            return BenchmarkImportSummary(processed=0, inserted=0, skipped=0)
        try:
            with self.engine.begin() as connection:
                result = connection.execute(statement, parameters)
        except SQLAlchemyError as exc:
            raise BenchmarkSourceError("failed to import benchmark batch") from exc
        inserted = max(0, min(len(parameters), result.rowcount))
        return BenchmarkImportSummary(
            processed=len(parameters),
            inserted=inserted,
            skipped=len(parameters) - inserted,
        )

    def list_cases(
        self,
        batch_name: str,
        personal_limit: int | None = None,
        non_personal_limit: int | None = None,
    ) -> list[BenchmarkCase]:
        """Return deterministic positive and negative cases without label leakage."""
        statement = text(
            """
            SELECT benchmark_id, batch_name, field_name,
                   sample_values_json, expected_personal
            FROM benchmark_field_input
            WHERE batch_name = :batch_name
            ORDER BY benchmark_id
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(
                statement,
                {"batch_name": batch_name},
            ).mappings().all()
        cases = [self._map_case(row) for row in rows]
        personal = [case for case in cases if case.expected_personal]
        non_personal = [case for case in cases if not case.expected_personal]
        if personal_limit is not None:
            personal = personal[:personal_limit]
        if non_personal_limit is not None:
            non_personal = non_personal[:non_personal_limit]
        return [*personal, *non_personal]

    @staticmethod
    def _map_case(row) -> BenchmarkCase:
        samples = row["sample_values_json"]
        if isinstance(samples, str):
            samples = json.loads(samples)
        return BenchmarkCase(
            benchmark_id=row["benchmark_id"],
            batch_name=row["batch_name"],
            expected_personal=bool(row["expected_personal"]),
            field_profile=FieldProfile(
                source_system="benchmark",
                database_name="teacher_benchmark",
                table_name="benchmark_input",
                field_name=row["field_name"],
                sample_values=samples,
                data_type="unknown",
                business_domain="general",
            ),
        )
