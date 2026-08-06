import json
from contextlib import nullcontext
from datetime import datetime, timezone
from uuid import UUID

from app.repositories.benchmark_source import BenchmarkSourceRepository
from app.repositories.benchmark_target import BenchmarkTargetRepository
from app.schemas.benchmark import (
    BenchmarkImportRow,
    BenchmarkPrediction,
    BenchmarkRunSummary,
)


class Result:
    def __init__(self, rowcount=2):
        self.rowcount = rowcount


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, parameters):
        self.calls.append((str(statement), parameters))
        return Result()


class RecordingEngine:
    def __init__(self):
        self.connection = RecordingConnection()
        self.begin_count = 0

    def begin(self):
        self.begin_count += 1
        return nullcontext(self.connection)


def test_source_repository_imports_both_labels_in_one_transaction():
    engine = RecordingEngine()
    repository = BenchmarkSourceRepository(engine=engine)
    rows = [
        BenchmarkImportRow(
            source_dataset="personal",
            source_row_number=2,
            field_name="email",
            sample_values=["a***@x.test"],
            expected_personal=True,
        ),
        BenchmarkImportRow(
            source_dataset="non_personal",
            source_row_number=2,
            field_name="created_at",
            sample_values=["2026-08-01"],
            expected_personal=False,
        ),
    ]

    summary = repository.import_batch("teacher_2026_08", rows)

    assert summary.inserted == 2
    assert summary.skipped == 0
    assert engine.begin_count == 1
    _, parameters = engine.connection.calls[0]
    assert len(parameters) == 2
    assert parameters[0]["expected_personal"] is True
    assert json.loads(parameters[1]["sample_values_json"]) == ["2026-08-01"]


def test_target_repository_maps_run_and_prediction_to_relational_parameters():
    engine = RecordingEngine()
    repository = BenchmarkTargetRepository(engine=engine)
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    summary = BenchmarkRunSummary(
        run_id=run_id,
        batch_name="teacher_2026_08",
        personal_limit=20,
        non_personal_limit=80,
        status="RUNNING",
        total_cases=100,
        model_name="fake",
        knowledge_base_version="v-test",
        source_type="csv",
        input_mode="tabular",
        source_name="business.csv",
        source_fingerprint="a" * 64,
        label_fingerprint="b" * 64,
        labeled_cases=80,
        unlabeled_cases=20,
        started_at=now,
    )
    prediction = BenchmarkPrediction(
        run_id=run_id,
        benchmark_id=7,
        field_name_snapshot="email",
        sample_values=["a***@x.test"],
        expected_personal=True,
        predicted_personal=True,
        outcome="TP",
        category="个人信息",
        level="L2",
        confidence=0.9,
        reason="test",
        need_review=False,
        decision_path="rag_llm",
        status="SUCCESS",
        created_at=now,
    )

    repository.create_run(summary)
    repository.save_prediction(prediction)

    run_params = engine.connection.calls[0][1]
    prediction_params = engine.connection.calls[1][1]
    assert run_params["personal_limit"] == 20
    assert run_params["source_type"] == "csv"
    assert run_params["input_mode"] == "tabular"
    assert run_params["labeled_cases"] == 80
    assert prediction_params["predicted_personal"] is True
    assert json.loads(prediction_params["sample_values_json"]) == ["a***@x.test"]


def test_target_repository_persists_unlabeled_csv_prediction():
    engine = RecordingEngine()
    repository = BenchmarkTargetRepository(engine=engine)
    prediction = BenchmarkPrediction(
        run_id=UUID("12345678-1234-5678-1234-567812345678"),
        benchmark_id=1,
        field_name_snapshot="created_at",
        expected_personal=None,
        predicted_personal=False,
        outcome="UNLABELED",
        category="业务信息",
        level="L1",
        confidence=0.8,
        reason="test",
        need_review=False,
        decision_path="rag_llm",
        status="SUCCESS",
        created_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )

    repository.save_prediction(prediction)

    parameters = engine.connection.calls[0][1]
    assert parameters["expected_personal"] is None
    assert parameters["outcome"] == "UNLABELED"
