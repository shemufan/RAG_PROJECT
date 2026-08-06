import csv
from datetime import datetime, timezone
from uuid import UUID

from app.schemas.classification import ClassificationResult
from app.services.benchmark_label_service import attach_benchmark_labels
from app.services.csv_pipeline import CSVClassificationPipeline
from scripts.run_csv_pipeline import load_csv_batch, parse_args

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


class FakeClassifier:
    def classify_field(self, profile):
        predicted = profile.field_name in {"email", "price"}
        return ClassificationResult(
            field_name=profile.field_name,
            is_personal=predicted,
            category="test",
            level="L2",
            confidence=0.9,
            reason="offline",
            need_review=False,
            decision_path="fake",
        )


class MemoryTarget:
    def __init__(self):
        self.predictions = []

    def create_run(self, summary):
        self.summary = summary

    def update_run(self, summary):
        self.summary = summary

    def save_prediction(self, prediction):
        self.predictions.append(prediction)

    def load_metric_inputs(self, run_id):
        return [row.metric_input() for row in self.predictions]


def test_catalog_and_tabular_reach_same_pipeline_without_external_calls(tmp_path):
    catalog = tmp_path / "catalog.csv"
    tabular = tmp_path / "business.csv"
    labels = tmp_path / "labels.csv"
    write_csv(catalog, [["字段名", "样本1"], ["email", "a***@x.test"]])
    write_csv(tabular, [["email", "price"], ["a***@x.test", "99"]])
    write_csv(
        labels,
        [["field_name", "expected_personal"], ["email", "true"], ["price", "false"]],
    )

    catalog_batch = load_csv_batch(parse_args(["--input", str(catalog)]))
    tabular_batch = load_csv_batch(parse_args(["--input", str(tabular)]))
    label_summary = attach_benchmark_labels(tabular_batch, labels)
    target = MemoryTarget()
    summary = CSVClassificationPipeline(
        target,
        FakeClassifier(),
        model_name="fake",
        knowledge_base_version="v-test",
        clock=lambda: NOW,
        run_id_factory=lambda: RUN_ID,
    ).run(tabular_batch, label_summary)

    assert catalog_batch.input_mode == "catalog"
    assert tabular_batch.input_mode == "tabular"
    assert catalog_batch.cases[0].field_profile.field_name == "email"
    assert (summary.tp, summary.fp, summary.tn, summary.fn) == (1, 1, 0, 0)
    assert summary.precision_score == 0.5
    assert summary.coverage_score == 1.0
