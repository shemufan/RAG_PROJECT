"""Synchronous orchestration from source database fields to compliance records."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.classification import ClassificationOutput
from app.schemas.pipeline import (
    FieldClassificationRecord,
    PipelineRequest,
    PipelineSummary,
    RunStatus,
    stable_field_id,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _final_status(total_fields: int, success_fields: int, failed_fields: int) -> RunStatus:
    if total_fields > 0 and success_fields == total_fields and failed_fields == 0:
        return "SUCCESS"
    if success_fields > 0:
        return "PARTIAL_FAILED"
    return "FAILED"


class DatabaseClassificationPipeline:
    """Coordinate source scanning, existing RAG classification, and target persistence."""

    def __init__(
        self,
        source_repository,
        target_repository,
        classification_service,
        *,
        model_name: str,
        knowledge_base_version: str,
        clock=_utc_now,
        run_id_factory=uuid4,
    ):
        self.source_repository = source_repository
        self.target_repository = target_repository
        self.classification_service = classification_service
        self.model_name = model_name
        self.knowledge_base_version = knowledge_base_version
        self.clock = clock
        self.run_id_factory = run_id_factory

    def run(self, request: PipelineRequest) -> PipelineSummary:
        """Execute one database classification run and return its final counters."""
        summary = PipelineSummary(
            run_id=self.run_id_factory(),
            source_database=self.source_repository.database_name,
            total_fields=0,
            success_fields=0,
            review_fields=0,
            failed_fields=0,
            status="RUNNING",
            started_at=self.clock(),
        )
        self.target_repository.create_run(
            summary,
            self.model_name,
            self.knowledge_base_version,
        )
        try:
            profiles = self.source_repository.scan_fields(
                sample_limit=request.sample_limit,
                table_names=request.table_names,
            )
        except Exception as exc:
            logger.exception("源数据库字段扫描失败")
            failed_summary = summary.model_copy(
                update={
                    "status": "FAILED",
                    "finished_at": self.clock(),
                }
            )
            self.target_repository.update_run(failed_summary, error_message=str(exc))
            return failed_summary

        success_fields = 0
        review_fields = 0
        failed_fields = 0
        for profile in profiles:
            field_id = stable_field_id(profile)
            seen_at = self.clock()
            try:
                self.target_repository.upsert_field_asset(field_id, profile, seen_at)
                result = self.classification_service.classify_field(profile)
                if result.level == "UNKNOWN":
                    raise RuntimeError("classification returned UNKNOWN")
                classification = ClassificationOutput(
                    category=result.category,
                    subcategory=result.subcategory,
                    level=result.level,
                    confidence=result.confidence,
                    reason=result.reason,
                    need_review=result.need_review,
                )
                record = FieldClassificationRecord(
                    run_id=summary.run_id,
                    field_id=field_id,
                    field_profile=profile,
                    classification=classification,
                    evidence=result.evidence,
                    decision_path=result.decision_path,
                    model_name=self.model_name,
                    knowledge_base_version=self.knowledge_base_version,
                    created_at=seen_at,
                )
                self.target_repository.save_classification_record(record)
            except Exception:
                failed_fields += 1
                logger.exception(
                    "字段分类或写入失败：%s.%s.%s",
                    profile.database_name,
                    profile.table_name,
                    profile.field_name,
                )
                if not request.continue_on_error:
                    break
                continue
            success_fields += 1
            if classification.need_review:
                review_fields += 1

        final_summary = summary.model_copy(
            update={
                "total_fields": len(profiles),
                "success_fields": success_fields,
                "review_fields": review_fields,
                "failed_fields": failed_fields,
                "status": _final_status(len(profiles), success_fields, failed_fields),
                "finished_at": self.clock(),
            }
        )
        self.target_repository.update_run(final_summary)
        return final_summary
