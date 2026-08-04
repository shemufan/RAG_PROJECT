from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.schemas.benchmark import BenchmarkCase
from app.schemas.classification import ClassificationResult
from app.schemas.field import FieldProfile
from app.services.benchmark_pipeline import BenchmarkClassificationPipeline
from scripts.run_benchmark import parse_args

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


def make_case(case_id, field_name, expected):
    return BenchmarkCase(
        benchmark_id=case_id,
        batch_name="teacher_2026_08",
        expected_personal=expected,
        field_profile=FieldProfile(
            source_system="benchmark",
            database_name="teacher_benchmark",
            table_name="benchmark_input",
            field_name=field_name,
            sample_values=["masked"],
        ),
    )


class FakeSource:
    def __init__(self, cases):
        self.cases = cases
        self.requests = []

    def list_cases(self, batch_name, personal_limit=None, non_personal_limit=None):
        self.requests.append((batch_name, personal_limit, non_personal_limit))
        personal = [case for case in self.cases if case.expected_personal]
        non_personal = [case for case in self.cases if not case.expected_personal]
        if personal_limit is not None:
            personal = personal[:personal_limit]
        if non_personal_limit is not None:
            non_personal = non_personal[:non_personal_limit]
        return [*personal, *non_personal]


class FakeClassifier:
    def __init__(self, predictions):
        self.predictions = predictions
        self.seen = []

    def classify_field(self, profile):
        self.seen.append(profile.field_name)
        value = self.predictions[profile.field_name]
        if isinstance(value, Exception):
            raise value
        return ClassificationResult(
            field_name=profile.field_name,
            is_personal=value,
            category="个人信息" if value else "业务信息",
            level="L2",
            confidence=0.9,
            reason="test",
            need_review=False,
            decision_path="rag_llm",
        )


class FakeTarget:
    def __init__(self):
        self.run = None
        self.predictions = []
        self.deleted = []

    def create_run(self, summary):
        self.run = summary

    def save_prediction(self, prediction):
        self.predictions.append(prediction)

    def load_metric_inputs(self, run_id):
        return [prediction.metric_input() for prediction in self.predictions]

    def update_run(self, summary):
        self.run = summary

    def get_run(self, run_id):
        return self.run if self.run and self.run.run_id == run_id else None

    def list_recorded_cases(self, run_id):
        return {
            prediction.benchmark_id: prediction.status
            for prediction in self.predictions
            if prediction.run_id == run_id
        }

    def delete_failed_prediction(self, run_id, benchmark_id):
        self.deleted.append(benchmark_id)
        self.predictions = [
            prediction
            for prediction in self.predictions
            if not (
                prediction.run_id == run_id
                and prediction.benchmark_id == benchmark_id
                and prediction.status == "FAILED"
            )
        ]


def test_pipeline_classifies_cases_persists_outcomes_and_scores():
    cases = [make_case(1, "email", True), make_case(2, "created_at", False)]
    source = FakeSource(cases)
    target = FakeTarget()
    pipeline = BenchmarkClassificationPipeline(
        source,
        target,
        FakeClassifier({"email": True, "created_at": True}),
        model_name="fake",
        knowledge_base_version="v-test",
        run_id_factory=lambda: RUN_ID,
        clock=lambda: NOW,
    )

    summary = pipeline.run("teacher_2026_08")

    assert summary.status == "SUCCESS"
    assert (summary.tp, summary.fp, summary.tn, summary.fn) == (1, 1, 0, 0)
    assert summary.precision_score == 0.5
    assert [prediction.outcome for prediction in target.predictions] == ["TP", "FP"]


def test_pipeline_records_failure_and_continues():
    cases = [make_case(1, "email", True), make_case(2, "created_at", False)]
    target = FakeTarget()
    pipeline = BenchmarkClassificationPipeline(
        FakeSource(cases),
        target,
        FakeClassifier({"email": RuntimeError("provider down"), "created_at": False}),
        model_name="fake",
        knowledge_base_version="v-test",
        run_id_factory=lambda: RUN_ID,
        clock=lambda: NOW,
    )

    summary = pipeline.run("teacher_2026_08")

    assert summary.status == "PARTIAL_FAILED"
    assert summary.failed_cases == 1
    assert target.predictions[0].outcome == "FAILED"
    assert target.predictions[0].predicted_personal is None
    assert target.predictions[1].outcome == "TN"


def test_pipeline_applies_positive_and_negative_limits_independently():
    cases = [
        make_case(1, "email", True),
        make_case(2, "phone", True),
        make_case(3, "created_at", False),
        make_case(4, "price", False),
    ]
    classifier = FakeClassifier({"email": True, "created_at": False, "price": False})
    target = FakeTarget()
    pipeline = BenchmarkClassificationPipeline(
        FakeSource(cases),
        target,
        classifier,
        model_name="fake",
        knowledge_base_version="v-test",
        run_id_factory=lambda: RUN_ID,
        clock=lambda: NOW,
    )

    summary = pipeline.run(
        "teacher_2026_08",
        personal_limit=1,
        non_personal_limit=2,
    )

    assert summary.total_cases == 3
    assert classifier.seen == ["email", "created_at", "price"]


def test_resume_skips_every_recorded_case_by_default():
    cases = [make_case(1, "email", True), make_case(2, "created_at", False)]
    target = FakeTarget()
    pipeline = BenchmarkClassificationPipeline(
        FakeSource(cases),
        target,
        FakeClassifier({"email": RuntimeError("down"), "created_at": False}),
        model_name="fake",
        knowledge_base_version="v-test",
        run_id_factory=lambda: RUN_ID,
        clock=lambda: NOW,
    )
    pipeline.run("teacher_2026_08")
    classifier = FakeClassifier({})
    pipeline.classification_service = classifier

    summary = pipeline.resume(RUN_ID)

    assert classifier.seen == []
    assert summary.failed_cases == 1


def test_resume_retry_failed_retries_only_failed_cases():
    cases = [make_case(1, "email", True), make_case(2, "created_at", False)]
    target = FakeTarget()
    pipeline = BenchmarkClassificationPipeline(
        FakeSource(cases),
        target,
        FakeClassifier({"email": RuntimeError("down"), "created_at": False}),
        model_name="fake",
        knowledge_base_version="v-test",
        run_id_factory=lambda: RUN_ID,
        clock=lambda: NOW,
    )
    pipeline.run("teacher_2026_08")
    classifier = FakeClassifier({"email": True})
    pipeline.classification_service = classifier

    summary = pipeline.resume(RUN_ID, retry_failed=True)

    assert classifier.seen == ["email"]
    assert target.deleted == [1]
    assert summary.failed_cases == 0
    assert summary.tp == 1


def test_cli_rejects_retry_failed_for_new_run():
    with pytest.raises(SystemExit):
        parse_args(["--batch", "teacher_2026_08", "--retry-failed"])


def test_cli_rejects_non_positive_limits():
    with pytest.raises(SystemExit):
        parse_args(["--batch", "teacher_2026_08", "--personal-limit", "0"])
