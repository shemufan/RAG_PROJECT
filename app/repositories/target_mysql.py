"""SQLAlchemy Core persistence for classification runs and relational results."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.field import FieldProfile
from app.schemas.pipeline import (
    ClassificationEvidenceRow,
    ClassificationResultRow,
    FieldClassificationRecord,
    PipelineSummary,
    RunDetail,
)


class TargetPersistenceError(RuntimeError):
    """Raised when a target database operation cannot be completed."""


class TargetMySQLRepository:
    """Persist and query compliance records without invoking classification services."""

    def __init__(self, database_url: str | None = None, *, engine=None):
        if engine is None:
            if not database_url:
                raise ValueError("TARGET_DATABASE_URL 未配置")
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine

    def create_run(
        self,
        summary: PipelineSummary,
        model_name: str,
        knowledge_version: str,
    ) -> None:
        statement = text(
            """
            INSERT INTO classification_run (
                run_id, source_system, source_database, status,
                total_fields, success_fields, review_fields, failed_fields,
                model_name, knowledge_base_version, started_at, finished_at, error_message
            ) VALUES (
                :run_id, 'mysql', :source_database, :status,
                :total_fields, :success_fields, :review_fields, :failed_fields,
                :model_name, :knowledge_base_version, :started_at, NULL, NULL
            )
            """
        )
        parameters = {
            "run_id": str(summary.run_id),
            "source_database": summary.source_database,
            "status": summary.status,
            "total_fields": summary.total_fields,
            "success_fields": summary.success_fields,
            "review_fields": summary.review_fields,
            "failed_fields": summary.failed_fields,
            "model_name": model_name,
            "knowledge_base_version": knowledge_version,
            "started_at": summary.started_at,
        }
        self._execute_write("create classification run", statement, parameters)

    def update_run(
        self,
        summary: PipelineSummary,
        error_message: str | None = None,
    ) -> None:
        statement = text(
            """
            UPDATE classification_run
            SET status = :status,
                total_fields = :total_fields,
                success_fields = :success_fields,
                review_fields = :review_fields,
                failed_fields = :failed_fields,
                finished_at = :finished_at,
                error_message = :error_message
            WHERE run_id = :run_id
            """
        )
        parameters = {
            "run_id": str(summary.run_id),
            "status": summary.status,
            "total_fields": summary.total_fields,
            "success_fields": summary.success_fields,
            "review_fields": summary.review_fields,
            "failed_fields": summary.failed_fields,
            "finished_at": summary.finished_at,
            "error_message": error_message,
        }
        self._execute_write("update classification run", statement, parameters)

    def upsert_field_asset(
        self,
        field_id: UUID,
        profile: FieldProfile,
        seen_at: datetime,
    ) -> None:
        statement = text(
            """
            INSERT INTO data_field_asset (
                field_id, source_system, database_name, table_name, table_comment,
                column_name, column_comment, data_type, is_nullable, column_key,
                business_domain, first_seen_at, last_seen_at
            ) VALUES (
                :field_id, :source_system, :database_name, :table_name, :table_comment,
                :column_name, :column_comment, :data_type, :is_nullable, :column_key,
                :business_domain, :first_seen_at, :last_seen_at
            )
            ON DUPLICATE KEY UPDATE
                table_comment = VALUES(table_comment),
                column_comment = VALUES(column_comment),
                data_type = VALUES(data_type),
                is_nullable = VALUES(is_nullable),
                column_key = VALUES(column_key),
                business_domain = VALUES(business_domain),
                last_seen_at = VALUES(last_seen_at)
            """
        )
        parameters = {
            "field_id": str(field_id),
            "source_system": profile.source_system,
            "database_name": profile.database_name,
            "table_name": profile.table_name,
            "table_comment": profile.table_comment,
            "column_name": profile.field_name,
            "column_comment": profile.field_comment,
            "data_type": profile.data_type,
            "is_nullable": profile.is_nullable,
            "column_key": profile.column_key,
            "business_domain": profile.business_domain,
            "first_seen_at": seen_at,
            "last_seen_at": seen_at,
        }
        self._execute_write("upsert field asset", statement, parameters)

    def save_classification_record(self, record: FieldClassificationRecord) -> int:
        result_statement = text(
            """
            INSERT INTO field_classification_result (
                run_id, field_id, category, subcategory, is_personal, level, confidence,
                reason, need_review, decision_path, input_snapshot_json,
                raw_output_json, created_at
            ) VALUES (
                :run_id, :field_id, :category, :subcategory, :is_personal, :level, :confidence,
                :reason, :need_review, :decision_path, :input_snapshot_json,
                :raw_output_json, :created_at
            )
            """
        )
        evidence_statement = text(
            """
            INSERT INTO classification_evidence (
                result_id, rank_no, document_name, article, content,
                relevance_score, chunk_id
            ) VALUES (
                :result_id, :rank_no, :document_name, :article, :content,
                :relevance_score, :chunk_id
            )
            """
        )
        result_parameters = self._result_parameters(record)
        try:
            with self.engine.begin() as connection:
                cursor = connection.execute(result_statement, result_parameters)
                result_id = cursor.lastrowid
                if result_id is None:
                    raise TargetPersistenceError("save classification record: missing result_id")
                evidence_parameters = [
                    {
                        "result_id": result_id,
                        "rank_no": rank,
                        "document_name": evidence.source,
                        "article": evidence.article,
                        "content": evidence.content,
                        "relevance_score": evidence.score,
                        "chunk_id": evidence.chunk_id,
                    }
                    for rank, evidence in enumerate(record.evidence, start=1)
                ]
                if evidence_parameters:
                    connection.execute(evidence_statement, evidence_parameters)
                return int(result_id)
        except TargetPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise TargetPersistenceError(f"save classification record: {exc}") from exc

    def get_run(self, run_id: UUID) -> RunDetail | None:
        statement = text(
            """
            SELECT run_id, source_system, source_database, status,
                   total_fields, success_fields, review_fields, failed_fields,
                   model_name, knowledge_base_version, started_at, finished_at, error_message
            FROM classification_run
            WHERE run_id = :run_id
            """
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement, {"run_id": str(run_id)}).mappings().first()
        return RunDetail.model_validate(dict(row)) if row else None

    def query_results(
        self,
        *,
        run_id: UUID | None = None,
        database_name: str | None = None,
        table_name: str | None = None,
        column_name: str | None = None,
        level: str | None = None,
        category: str | None = None,
        need_review: bool | None = None,
        is_personal: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ClassificationResultRow]:
        filters = []
        parameters: dict[str, Any] = {"limit": limit, "offset": offset}
        optional_filters = {
            "r.run_id": ("run_id", str(run_id) if run_id else None),
            "a.database_name": ("database_name", database_name),
            "a.table_name": ("table_name", table_name),
            "a.column_name": ("column_name", column_name),
            "r.level": ("level", level),
            "r.category": ("category", category),
            "r.need_review": ("need_review", need_review),
            "r.is_personal": ("is_personal", is_personal),
        }
        for column, (name, value) in optional_filters.items():
            if value is not None:
                filters.append(f"{column} = :{name}")
                parameters[name] = value
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        statement = text(
            f"""
            SELECT r.result_id, r.run_id, r.field_id,
                   a.source_system, a.database_name, a.table_name,
                   a.column_name, a.business_domain,
                   r.category, r.subcategory, r.is_personal, r.level, r.confidence,
                   r.reason, r.need_review, r.decision_path, r.created_at
            FROM field_classification_result AS r
            JOIN data_field_asset AS a ON a.field_id = r.field_id
            {where_clause}
            ORDER BY r.created_at DESC, r.result_id DESC
            LIMIT :limit OFFSET :offset
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, parameters).mappings().all()
        return [ClassificationResultRow.model_validate(dict(row)) for row in rows]

    def get_result_evidence(self, result_id: int) -> list[ClassificationEvidenceRow]:
        statement = text(
            """
            SELECT evidence_id, result_id, rank_no, document_name, article,
                   content, relevance_score, chunk_id
            FROM classification_evidence
            WHERE result_id = :result_id
            ORDER BY rank_no ASC
            """
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"result_id": result_id}).mappings().all()
        return [ClassificationEvidenceRow.model_validate(dict(row)) for row in rows]

    def _execute_write(self, operation: str, statement, parameters: dict[str, Any]) -> None:
        try:
            with self.engine.begin() as connection:
                connection.execute(statement, parameters)
        except SQLAlchemyError as exc:
            raise TargetPersistenceError(f"{operation}: {exc}") from exc

    @staticmethod
    def _result_parameters(record: FieldClassificationRecord) -> dict[str, Any]:
        classification = record.classification
        return {
            "run_id": str(record.run_id),
            "field_id": str(record.field_id),
            "category": classification.category,
            "subcategory": classification.subcategory,
            "is_personal": classification.is_personal,
            "level": classification.level,
            "confidence": classification.confidence,
            "reason": classification.reason,
            "need_review": classification.need_review,
            "decision_path": record.decision_path,
            "input_snapshot_json": json.dumps(
                record.field_profile.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "raw_output_json": json.dumps(
                classification.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "created_at": record.created_at,
        }
