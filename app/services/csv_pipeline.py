"""Direct CSV-to-RAG-to-B classification orchestration."""

from datetime import datetime, timezone
from uuid import uuid4

from app.schemas.benchmark import BenchmarkPrediction, BenchmarkRunSummary
from app.schemas.csv_input import CSVFieldCase, CSVInputBatch, LabelMatchSummary
from app.services.benchmark_evaluator import evaluate_predictions


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _outcome(expected: bool | None, predicted: bool) -> str:
    if expected is None:
        return "UNLABELED"
    if expected:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


class CSVClassificationPipeline:
    """Classify canonical CSV cases without requiring database A."""

    def __init__(
        self,
        target_repository,
        classification_service,
        *,
        model_name: str,
        knowledge_base_version: str,
        clock=_utc_now,
        run_id_factory=uuid4,
    ):
        self.target_repository = target_repository
        self.classification_service = classification_service
        self.model_name = model_name
        self.knowledge_base_version = knowledge_base_version
        self.clock = clock
        self.run_id_factory = run_id_factory

    def run(
        self,
        batch: CSVInputBatch,
        labels: LabelMatchSummary | None = None,
    ) -> BenchmarkRunSummary:
        cases = self._select_cases(batch, labels)
        if not cases:
            raise ValueError("CSV input contains no fields")
        summary = BenchmarkRunSummary(
            run_id=self.run_id_factory(),
            batch_name=batch.source_name[:64],
            personal_limit=batch.personal_limit,
            non_personal_limit=batch.non_personal_limit,
            source_type="csv",
            input_mode=batch.input_mode,
            source_name=batch.source_name,
            source_fingerprint=batch.source_fingerprint,
            label_fingerprint=labels.label_fingerprint if labels else None,
            status="RUNNING",
            total_cases=len(cases),
            labeled_cases=sum(case.expected_personal is not None for case in cases),
            unlabeled_cases=sum(case.expected_personal is None for case in cases),
            model_name=self.model_name,
            knowledge_base_version=self.knowledge_base_version,
            started_at=self.clock(),
        )
        self.target_repository.create_run(summary)
        return self._execute(summary, cases)

    def resume(
        self,
        run_id,
        batch: CSVInputBatch,
        labels: LabelMatchSummary | None = None,
        *,
        retry_failed: bool = False,
    ) -> BenchmarkRunSummary:
        summary = self.target_repository.get_run(run_id)
        if summary is None or summary.source_type != "csv":
            raise ValueError("CSV benchmark run not found")
        if (
            summary.model_name != self.model_name
            or summary.knowledge_base_version != self.knowledge_base_version
        ):
            raise ValueError("resume classification configuration does not match original run")
        cases = self._select_cases(batch, labels)
        label_fingerprint = labels.label_fingerprint if labels else None
        if (
            summary.source_fingerprint != batch.source_fingerprint
            or summary.input_mode != batch.input_mode
            or summary.label_fingerprint != label_fingerprint
        ):
            raise ValueError("resume input does not match the original run")

        if len(cases) < summary.total_cases:
            raise ValueError("resume case range does not match the original run")
        cases = cases[:summary.total_cases]

        recorded = self.target_repository.list_recorded_case_snapshots(run_id)
        selected = []
        for case in cases:
            snapshot = recorded.get(case.case_index)
            if snapshot is None:
                selected.append(case)
                continue
            field_name, status = snapshot
            if field_name != case.field_profile.field_name:
                raise ValueError("resume case mapping does not match the original run")
            if retry_failed and status == "FAILED":
                self.target_repository.delete_failed_prediction(run_id, case.case_index)
                selected.append(case)
        running = summary.model_copy(
            update={"status": "RUNNING", "finished_at": None, "error_message": None}
        )
        self.target_repository.update_run(running)
        return self._execute(running, selected)

    @staticmethod
    def _select_cases(
        batch: CSVInputBatch,
        labels: LabelMatchSummary | None,
    ) -> list[CSVFieldCase]:
        if labels is None:
            return batch.cases
        batch_mapping = [
            (case.case_index, case.field_profile.field_name) for case in batch.cases
        ]
        label_mapping = [
            (case.case_index, case.field_profile.field_name) for case in labels.cases
        ]
        if batch_mapping != label_mapping:
            raise ValueError("label cases do not match CSV input cases")
        return labels.cases

    def _execute(
        self,
        summary: BenchmarkRunSummary,
        cases: list[CSVFieldCase],
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

    def _classify_case(self, run_id, case: CSVFieldCase):
        profile = case.field_profile
        try:
            result = self.classification_service.classify_field(profile)
            if result.level == "UNKNOWN" or result.is_personal is None:
                raise RuntimeError("classification returned UNKNOWN")
            prediction = BenchmarkPrediction(
                run_id=run_id,
                benchmark_id=case.case_index,
                field_name_snapshot=profile.field_name,
                sample_values=profile.sample_values,
                expected_personal=case.expected_personal,
                predicted_personal=result.is_personal,
                outcome=_outcome(case.expected_personal, result.is_personal),
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
            prediction = BenchmarkPrediction(
                run_id=run_id,
                benchmark_id=case.case_index,
                field_name_snapshot=profile.field_name,
                sample_values=profile.sample_values,
                expected_personal=case.expected_personal,
                outcome="FAILED",
                status="FAILED",
                error_message=type(exc).__name__,
                created_at=self.clock(),
            )
        return prediction
