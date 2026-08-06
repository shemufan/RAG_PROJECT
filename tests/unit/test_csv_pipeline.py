from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.schemas.classification import ClassificationResult
from app.schemas.csv_input import CSVFieldCase, CSVInputBatch, LabelMatchSummary
from app.schemas.field import FieldProfile
from app.services.csv_pipeline import CSVClassificationPipeline

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def case(index, name, expected=None):
    return CSVFieldCase(
        case_index=index,
        expected_personal=expected,
        field_profile=FieldProfile(field_name=name, sample_values=["masked"]),
    )


def batch(*cases, fingerprint="a" * 64, mode="tabular"):
    return CSVInputBatch(
        source_name="business.csv",
        source_fingerprint=fingerprint,
        input_mode=mode,
        cases=list(cases),
    )


class FakeClassifier:
    def __init__(self, predictions):
        self.predictions = predictions
        self.seen = []

    def classify_field(self, profile):
        assert isinstance(profile, FieldProfile)
        assert not hasattr(profile, "expected_personal")
        self.seen.append(profile.field_name)
        value = self.predictions[profile.field_name]
        if isinstance(value, Exception):
            raise value
        return ClassificationResult(
            field_name=profile.field_name,
            is_personal=value,
            category="test",
            level="L2",
            confidence=0.9,
            reason="test",
            need_review=False,
            decision_path="fake",
        )


class FakeTarget:
    def __init__(self):
        self.run = None
        self.predictions = []
        self.deleted = []

    def create_run(self, summary):
        self.run = summary

    def update_run(self, summary):
        self.run = summary

    def save_prediction(self, prediction):
        self.predictions = [
            row
            for row in self.predictions
            if not (
                row.run_id == prediction.run_id
                and row.benchmark_id == prediction.benchmark_id
            )
        ]
        self.predictions.append(prediction)

    def load_metric_inputs(self, run_id):
        return [row.metric_input() for row in self.predictions if row.run_id == run_id]

    def get_run(self, run_id):
        return self.run if self.run and self.run.run_id == run_id else None

    def list_recorded_case_snapshots(self, run_id):
        return {
            row.benchmark_id: (row.field_name_snapshot, row.status)
            for row in self.predictions
            if row.run_id == run_id
        }

    def delete_failed_prediction(self, run_id, benchmark_id):
        self.deleted.append(benchmark_id)
        self.predictions = [
            row
            for row in self.predictions
            if not (
                row.run_id == run_id
                and row.benchmark_id == benchmark_id
                and row.status == "FAILED"
            )
        ]


def make_pipeline(target, classifier):
    return CSVClassificationPipeline(
        target,
        classifier,
        model_name="fake",
        knowledge_base_version="v-test",
        clock=lambda: NOW,
        run_id_factory=lambda: RUN_ID,
    )


def test_csv_pipeline_scores_labels_and_marks_unlabeled_results():
    target = FakeTarget()
    classifier = FakeClassifier({"email": True, "price": False})
    source = batch(case(1, "email"), case(2, "price"))
    labels = LabelMatchSummary(
        label_fingerprint="b" * 64,
        labeled_cases=1,
        unlabeled_cases=1,
        cases=[case(1, "email", True), case(2, "price", None)],
    )

    summary = make_pipeline(target, classifier).run(source, labels)

    assert summary.status == "SUCCESS"
    assert summary.source_type == "csv"
    assert summary.input_mode == "tabular"
    assert summary.tp == 1
    assert summary.labeled_cases == 1
    assert [row.outcome for row in target.predictions] == ["TP", "UNLABELED"]
    assert classifier.seen == ["email", "price"]


def test_csv_pipeline_records_unknown_and_continues():
    target = FakeTarget()
    classifier = FakeClassifier({"email": RuntimeError("down"), "price": False})

    summary = make_pipeline(target, classifier).run(
        batch(case(1, "email"), case(2, "price"))
    )

    assert summary.status == "PARTIAL_FAILED"
    assert [row.outcome for row in target.predictions] == ["FAILED", "UNLABELED"]
    assert classifier.seen == ["email", "price"]


@pytest.mark.parametrize(
    "changed",
    [
        batch(case(1, "email"), fingerprint="c" * 64),
        batch(case(1, "email"), mode="catalog"),
    ],
)
def test_resume_rejects_changed_source_or_mode(changed):
    target = FakeTarget()
    pipeline = make_pipeline(target, FakeClassifier({"email": True}))
    pipeline.run(batch(case(1, "email")))

    with pytest.raises(ValueError, match="does not match"):
        pipeline.resume(RUN_ID, changed)


def test_resume_skips_success_and_retries_only_failed():
    target = FakeTarget()
    pipeline = make_pipeline(
        target,
        FakeClassifier({"email": RuntimeError("down"), "price": False}),
    )
    source = batch(case(1, "email"), case(2, "price"))
    pipeline.run(source)
    classifier = FakeClassifier({"email": True})
    pipeline.classification_service = classifier

    summary = pipeline.resume(RUN_ID, source, retry_failed=True)

    assert classifier.seen == ["email"]
    assert target.deleted == [1]
    assert summary.failed_cases == 0


def test_resume_rejects_changed_recorded_field_name():
    target = FakeTarget()
    pipeline = make_pipeline(target, FakeClassifier({"email": True}))
    source = batch(case(1, "email"))
    pipeline.run(source)
    target.predictions[0].field_name_snapshot = "different"

    with pytest.raises(ValueError, match="case mapping"):
        pipeline.resume(RUN_ID, source)
