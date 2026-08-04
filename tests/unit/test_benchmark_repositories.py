import json
from contextlib import nullcontext

from app.repositories.benchmark_source import BenchmarkSourceRepository
from app.schemas.benchmark import BenchmarkImportRow


class Result:
    rowcount = 2


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

    imported = repository.import_batch("teacher_2026_08", rows)

    assert imported == 2
    assert engine.begin_count == 1
    _, parameters = engine.connection.calls[0]
    assert len(parameters) == 2
    assert parameters[0]["expected_personal"] is True
    assert json.loads(parameters[1]["sample_values_json"]) == ["2026-08-01"]
