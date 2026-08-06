"""B-database persistence and queries for benchmark executions."""

import json
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.benchmark import (
    BenchmarkMetricInput,
    BenchmarkPrediction,
    BenchmarkPredictionRow,
    BenchmarkRunSummary,
)


class BenchmarkTargetError(RuntimeError):
    """Raised when benchmark result persistence fails."""


class BenchmarkTargetRepository:
    """Persist benchmark runs independently from production field assets."""

    def __init__(self, database_url: str | None = None, *, engine=None):
        if engine is None:
            if not database_url:
                raise ValueError("TARGET_DATABASE_URL is not configured")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine

    def create_run(self, summary: BenchmarkRunSummary) -> None:
        statement = text(
            """
            INSERT INTO benchmark_run (
                run_id, batch_name, personal_limit, non_personal_limit,
                source_type, input_mode, source_name, source_fingerprint,
                label_fingerprint, status, total_cases, success_cases,
                failed_cases, labeled_cases, unlabeled_cases, tp, fp, tn, fn,
                precision_score, recall_score, f1_score, accuracy_score,
                coverage_score, effective_recall_score, model_name,
                knowledge_base_version, started_at, finished_at, error_message
            ) VALUES (
                :run_id, :batch_name, :personal_limit, :non_personal_limit,
                :source_type, :input_mode, :source_name, :source_fingerprint,
                :label_fingerprint, :status, :total_cases, :success_cases,
                :failed_cases, :labeled_cases, :unlabeled_cases, :tp, :fp, :tn, :fn,
                :precision_score, :recall_score, :f1_score, :accuracy_score,
                :coverage_score, :effective_recall_score, :model_name,
                :knowledge_base_version, :started_at, :finished_at, :error_message
            )
            """
        )
        self._write("create benchmark run", statement, self._run_parameters(summary))

    def update_run(self, summary: BenchmarkRunSummary) -> None:
        statement = text(
            """
            UPDATE benchmark_run SET
                status=:status, total_cases=:total_cases,
                success_cases=:success_cases, failed_cases=:failed_cases,
                labeled_cases=:labeled_cases, unlabeled_cases=:unlabeled_cases,
                tp=:tp, fp=:fp, tn=:tn, fn=:fn,
                precision_score=:precision_score, recall_score=:recall_score,
                f1_score=:f1_score, accuracy_score=:accuracy_score,
                coverage_score=:coverage_score,
                effective_recall_score=:effective_recall_score,
                finished_at=:finished_at, error_message=:error_message
            WHERE run_id=:run_id
            """
        )
        self._write("update benchmark run", statement, self._run_parameters(summary))

    def save_prediction(self, prediction: BenchmarkPrediction) -> None:
        statement = text(
            """
            INSERT INTO benchmark_prediction (
                run_id, benchmark_id, field_name_snapshot, sample_values_json,
                expected_personal, predicted_personal, outcome, category,
                subcategory, level, confidence, reason, need_review,
                decision_path, evidence_json, status, error_message, created_at
            ) VALUES (
                :run_id, :benchmark_id, :field_name_snapshot, :sample_values_json,
                :expected_personal, :predicted_personal, :outcome, :category,
                :subcategory, :level, :confidence, :reason, :need_review,
                :decision_path, :evidence_json, :status, :error_message, :created_at
            )
            ON DUPLICATE KEY UPDATE
                predicted_personal=VALUES(predicted_personal), outcome=VALUES(outcome),
                category=VALUES(category), subcategory=VALUES(subcategory),
                level=VALUES(level), confidence=VALUES(confidence), reason=VALUES(reason),
                need_review=VALUES(need_review), decision_path=VALUES(decision_path),
                evidence_json=VALUES(evidence_json), status=VALUES(status),
                error_message=VALUES(error_message), created_at=VALUES(created_at)
            """
        )
        parameters = prediction.model_dump(exclude={"sample_values", "evidence"})
        parameters["run_id"] = str(prediction.run_id)
        parameters["sample_values_json"] = json.dumps(
            prediction.sample_values, ensure_ascii=False, separators=(",", ":")
        )
        parameters["evidence_json"] = json.dumps(
            [item.model_dump(mode="json") for item in prediction.evidence],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._write("save benchmark prediction", statement, parameters)

    def load_metric_inputs(self, run_id: UUID) -> list[BenchmarkMetricInput]:
        statement = text(
            """
            SELECT expected_personal, predicted_personal
            FROM benchmark_prediction WHERE run_id=:run_id
            ORDER BY benchmark_id
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"run_id": str(run_id)}).mappings().all()
        return [BenchmarkMetricInput.model_validate(dict(row)) for row in rows]

    def list_recorded_case_snapshots(self, run_id: UUID) -> dict[int, tuple[str, str]]:
        statement = text(
            """
            SELECT benchmark_id, field_name_snapshot, status
            FROM benchmark_prediction WHERE run_id=:run_id
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"run_id": str(run_id)}).mappings().all()
        return {
            int(row["benchmark_id"]): (
                str(row["field_name_snapshot"]),
                str(row["status"]),
            )
            for row in rows
        }

    def delete_failed_prediction(self, run_id: UUID, benchmark_id: int) -> None:
        statement = text(
            """
            DELETE FROM benchmark_prediction
            WHERE run_id=:run_id AND benchmark_id=:benchmark_id AND status='FAILED'
            """
        )
        self._write(
            "delete failed benchmark prediction",
            statement,
            {"run_id": str(run_id), "benchmark_id": benchmark_id},
        )

    def get_run(self, run_id: UUID) -> BenchmarkRunSummary | None:
        statement = text("SELECT * FROM benchmark_run WHERE run_id=:run_id")
        with self.engine.connect() as connection:
            row = connection.execute(statement, {"run_id": str(run_id)}).mappings().first()
        return BenchmarkRunSummary.model_validate(dict(row)) if row else None

    def query_predictions(
        self,
        *,
        run_id: UUID,
        outcome: str | None = None,
        predicted_personal: bool | None = None,
        need_review: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[BenchmarkPredictionRow]:
        filters = ["run_id=:run_id"]
        parameters = {"run_id": str(run_id), "limit": limit, "offset": offset}
        for column, value in (
            ("outcome", outcome),
            ("predicted_personal", predicted_personal),
            ("need_review", need_review),
        ):
            if value is not None:
                filters.append(f"{column}=:{column}")
                parameters[column] = value
        statement = text(
            "SELECT * FROM benchmark_prediction WHERE "
            + " AND ".join(filters)
            + " ORDER BY benchmark_id LIMIT :limit OFFSET :offset"
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, parameters).mappings().all()
        return [self._map_prediction_row(row) for row in rows]

    def _write(self, operation: str, statement, parameters) -> None:
        try:
            with self.engine.begin() as connection:
                connection.execute(statement, parameters)
        except SQLAlchemyError as exc:
            raise BenchmarkTargetError(operation) from exc

    @staticmethod
    def _run_parameters(summary: BenchmarkRunSummary) -> dict:
        values = summary.model_dump()
        values["run_id"] = str(summary.run_id)
        return values

    @staticmethod
    def _map_prediction_row(row) -> BenchmarkPredictionRow:
        values = dict(row)
        sample_values = values.pop("sample_values_json", [])
        evidence = values.pop("evidence_json", [])
        if isinstance(sample_values, str):
            sample_values = json.loads(sample_values)
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        values["sample_values"] = sample_values or []
        values["evidence"] = evidence or []
        return BenchmarkPredictionRow.model_validate(values)
