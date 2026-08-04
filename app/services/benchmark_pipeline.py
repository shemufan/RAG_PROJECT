"""Sequential A-to-RAG-to-B benchmark orchestration."""

from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.benchmark import (
    BenchmarkCase,
    BenchmarkPrediction,
    BenchmarkRunSummary,
)
from app.services.benchmark_evaluator import evaluate_predictions


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _outcome(expected: bool, predicted: bool) -> str:
    if expected:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


class BenchmarkClassificationPipeline:
    """Classify stored benchmark rows while keeping labels outside model input."""

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

    def run(
        self,
        batch_name: str,
        *,
        personal_limit: int | None = None,
        non_personal_limit: int | None = None,
    ) -> BenchmarkRunSummary:
        cases = self.source_repository.list_cases(
            batch_name,
            personal_limit=personal_limit,
            non_personal_limit=non_personal_limit,
        )
        summary = BenchmarkRunSummary(
            run_id=self.run_id_factory(),
            batch_name=batch_name,
            personal_limit=personal_limit,
            non_personal_limit=non_personal_limit,
            status="RUNNING",
            total_cases=len(cases),
            model_name=self.model_name,
            knowledge_base_version=self.knowledge_base_version,
            started_at=self.clock(),
        )
        self.target_repository.create_run(summary)
        return self._execute(summary, cases)

    def resume(
        self,
        run_id,
        *,
        retry_failed: bool = False,
    ) -> BenchmarkRunSummary:
        """Continue missing cases and optionally replace only failed predictions."""
        summary = self.target_repository.get_run(run_id)
        if summary is None:
            raise ValueError("benchmark run not found")
        cases = self.source_repository.list_cases(
            summary.batch_name,
            personal_limit=summary.personal_limit,
            non_personal_limit=summary.non_personal_limit,
        )
        recorded = self.target_repository.list_recorded_cases(run_id)
        selected: list[BenchmarkCase] = []
        for case in cases:
            status = recorded.get(case.benchmark_id)
            if status is None:
                selected.append(case)
            elif retry_failed and status == "FAILED":
                self.target_repository.delete_failed_prediction(
                    run_id,
                    case.benchmark_id,
                )
                selected.append(case)
        running = summary.model_copy(
            update={"status": "RUNNING", "finished_at": None, "error_message": None}
        )
        self.target_repository.update_run(running)
        return self._execute(running, selected)

    def _execute(
        self,
        summary: BenchmarkRunSummary,
        cases: list[BenchmarkCase],
    ) -> BenchmarkRunSummary:
        for case in cases:
            prediction = self._classify_case(summary.run_id, case)
            self.target_repository.save_prediction(prediction)

        metrics = evaluate_predictions(
            self.target_repository.load_metric_inputs(summary.run_id)
        )
        if metrics.failed_cases == 0:
            status = "SUCCESS"
        elif metrics.success_cases:
            status = "PARTIAL_FAILED"
        else:
            status = "FAILED"
        finished = summary.model_copy(
            update={
                **metrics.model_dump(),
                "status": status,
                "finished_at": self.clock(),
            }
        )
        self.target_repository.update_run(finished)
        return finished

    def _classify_case(self, run_id, case: BenchmarkCase) -> BenchmarkPrediction:
        profile = case.field_profile
        try:
            result = self.classification_service.classify_field(profile)
            if result.level == "UNKNOWN" or result.is_personal is None:
                raise RuntimeError("classification returned UNKNOWN")
            predicted = result.is_personal
            return BenchmarkPrediction(
                run_id=run_id,
                benchmark_id=case.benchmark_id,
                field_name_snapshot=profile.field_name,
                sample_values=profile.sample_values,
                expected_personal=case.expected_personal,
                predicted_personal=predicted,
                outcome=_outcome(case.expected_personal, predicted),
                category=result.category,
                subcategory=result.subcategory,
                level=result.level,
                confidence=result.confidence,
                reason=result.reason,
                need_review=result.need_review,
                decision_path=result.decision_path,
                evidence=result.evidence,
                status="SUCCESS",
                created_at=self.clock(),
            )
        except Exception as exc:
            return BenchmarkPrediction(
                run_id=run_id,
                benchmark_id=case.benchmark_id,
                field_name_snapshot=profile.field_name,
                sample_values=profile.sample_values,
                expected_personal=case.expected_personal,
                outcome="FAILED",
                status="FAILED",
                error_message=type(exc).__name__,
                created_at=self.clock(),
            )
