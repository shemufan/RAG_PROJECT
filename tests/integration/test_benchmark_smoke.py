from datetime import datetime, timezone
from uuid import UUID

from app.schemas.classification import ClassificationResult
from app.services.benchmark_label_service import prepare_labeled_catalog_benchmark
from app.services.csv_pipeline import CSVClassificationPipeline

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 4, tzinfo=timezone.utc)


class MemoryTarget:
    def __init__(self):
        self.predictions = []

    def create_run(self, summary):
        self.summary = summary

    def save_prediction(self, prediction):
        self.predictions.append(prediction)

    def load_metric_inputs(self, run_id):
        return [prediction.metric_input() for prediction in self.predictions]

    def update_run(self, summary):
        self.summary = summary


class FakeClassifier:
    predictions = {"personal_tp": True, "personal_fn": False, "other_fp": True, "other_tn": False}

    def classify_field(self, profile):
        predicted = self.predictions[profile.field_name]
        return ClassificationResult(
            field_name=profile.field_name,
            is_personal=predicted,
            category="test",
            level="L2",
            confidence=0.9,
            reason="offline smoke test",
            need_review=False,
            decision_path="fake",
        )


def test_generated_csv_to_classification_to_metrics_without_external_calls(tmp_path):
    personal_path = tmp_path / "personal.csv"
    non_personal_path = tmp_path / "non_personal.csv"
    personal_path.write_text(
        "字段名,样本1\npersonal_tp,masked-a\npersonal_fn,masked-b\n",
        encoding="utf-8-sig",
    )
    non_personal_path.write_text(
        "字段名,样本1\nother_fp,masked-c\nother_tn,masked-d\n",
        encoding="utf-8-sig",
    )
    prepared = prepare_labeled_catalog_benchmark(
        personal_path,
        non_personal_path,
        batch_name="smoke",
    )
    target = MemoryTarget()
    pipeline = CSVClassificationPipeline(
        target,
        FakeClassifier(),
        model_name="fake",
        knowledge_base_version="v-test",
        clock=lambda: NOW,
        run_id_factory=lambda: RUN_ID,
    )

    summary = pipeline.run(prepared.batch, prepared.labels)

    assert (summary.tp, summary.fp, summary.tn, summary.fn) == (1, 1, 1, 1)
    assert summary.precision_score == 0.5
    assert summary.recall_score == 0.5
    assert summary.f1_score == 0.5
    assert summary.coverage_score == 1.0
    assert summary.source_type == "csv"
