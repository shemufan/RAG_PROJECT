import csv
import json
from datetime import datetime, timezone
from uuid import UUID

from app.schemas.benchmark import BenchmarkPrediction, BenchmarkRunSummary
from app.schemas.classification import (
    ClassificationResult,
    Evidence,
    RegulationRetrievalResult,
    SemanticBridgeTrace,
)
from app.schemas.csv_input import CSVFieldCase
from app.schemas.field import FieldProfile
from app.schemas.semantic import (
    ObjectiveValueProfile,
    SemanticCard,
    SemanticRetrievalResult,
)
from app.services.experiment_e_reporter import ExperimentEReporter

RUN_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def _summary() -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        run_id=RUN_ID,
        batch_name="benchmark.csv",
        source_type="csv",
        input_mode="catalog",
        source_name="benchmark.csv",
        status="SUCCESS",
        total_cases=1,
        success_cases=1,
        labeled_cases=1,
        tp=1,
        precision_score=1.0,
        recall_score=1.0,
        f1_score=1.0,
        accuracy_score=1.0,
        coverage_score=1.0,
        effective_recall_score=1.0,
        model_name="same-as-b",
        knowledge_base_version="v1",
        started_at=NOW,
        finished_at=NOW,
    )


def _case_result_prediction():
    field = FieldProfile(
        field_name="contact_value",
        sample_values=["13812345678", "15987654321"],
    )
    card = SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号"],
        common_field_names=["phone"],
        value_features=["11位数字"],
        description="用于联系自然人的电话号码",
        semantic_category=["联系方式"],
        regulation_keywords=["电话号码"],
    )
    trace = SemanticBridgeTrace(
        profiling=ObjectiveValueProfile(features=["字符串长度约11位"]),
        semantic_query="semantic query",
        semantic_retrieval=[
            SemanticRetrievalResult(card=card, raw_score=0.82),
            SemanticRetrievalResult(
                card=card.model_copy(update={"semantic_type": "用户ID"}),
                raw_score=0.61,
            ),
        ],
        selected_semantic_type="手机号码",
        selected_card=card,
        top1_top2_score_gap=0.21,
        regulation_query="regulation query",
        regulation_retrieval=[
            RegulationRetrievalResult(
                evidence=Evidence(
                    content="联系方式属于个人信息",
                    source="rules.md",
                    chunk_id="chunk-1",
                    score=0.71,
                ),
                raw_score=0.71,
            )
        ],
    )
    result = ClassificationResult(
        field_name=field.field_name,
        is_personal=True,
        category="个人基本资料",
        subcategory="手机号码",
        level="L3",
        confidence=0.9,
        reason="命中联系方式规则",
        need_review=False,
        decision_path="semantic_rag_llm",
        evidence=[item.evidence for item in trace.regulation_retrieval],
        experiment_trace=trace,
    )
    prediction = BenchmarkPrediction(
        run_id=RUN_ID,
        benchmark_id=1,
        field_name_snapshot=field.field_name,
        sample_values=field.sample_values,
        expected_personal=True,
        predicted_personal=True,
        outcome="TP",
        category=result.category,
        subcategory=result.subcategory,
        level=result.level,
        confidence=result.confidence,
        reason=result.reason,
        need_review=False,
        decision_path=result.decision_path,
        evidence=result.evidence,
        status="SUCCESS",
        created_at=NOW,
    )
    case = CSVFieldCase(case_index=1, field_profile=field, expected_personal=True)
    return case, result, prediction


def test_reporter_writes_three_run_isolated_utf8_bom_artifacts(tmp_path):
    reporter = ExperimentEReporter(
        tmp_path,
        parameters={"semantic_top_k": 3, "regulation_top_k": 3},
    )
    summary = _summary()
    case, result, prediction = _case_result_prediction()

    reporter.start_run(summary)
    reporter.record_case(case, result, prediction)
    paths = reporter.finalize(summary)

    output_dir = tmp_path / "experiment_E" / str(RUN_ID)
    assert paths.results == output_dir / "experiment_E_results.csv"
    assert paths.summary == output_dir / "experiment_E_summary.json"
    assert paths.semantic_debug == output_dir / "semantic_retrieval_debug.csv"
    assert all(path.is_file() for path in paths)
    assert paths.results.read_bytes().startswith(b"\xef\xbb\xbf")

    with paths.results.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["field_name"] == "contact_value"
    assert json.loads(row["profiling"])["features"] == ["字符串长度约11位"]
    assert json.loads(row["regulation_retrieval"])[0]["raw_score"] == 0.71
    assert row["selected_semantic_type"] == "手机号码"
    assert row["correct"] == "True"

    with paths.semantic_debug.open(encoding="utf-8-sig", newline="") as handle:
        debug = next(csv.DictReader(handle))
    assert debug["top1_semantic_type"] == "手机号码"
    assert debug["top1_raw_score"] == "0.82"
    assert debug["top2_semantic_type"] == "用户ID"
    assert debug["top1_top2_score_gap"] == "0.21"

    payload = json.loads(paths.summary.read_text(encoding="utf-8-sig"))
    assert payload["experiment"] == "E"
    assert payload["metrics"]["tp"] == 1
    assert payload["parameters"]["semantic_top_k"] == 3


def test_reporter_persists_each_case_and_replaces_it_when_resumed(tmp_path):
    summary = _summary()
    case, result, prediction = _case_result_prediction()
    parameters = {"semantic_top_k": 3, "regulation_top_k": 3}
    first = ExperimentEReporter(tmp_path, parameters=parameters)

    first.start_run(summary)
    first.record_case(case, result, prediction)

    assert first.paths is not None
    assert first.paths.results.is_file()
    with first.paths.results.open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1

    resumed = ExperimentEReporter(tmp_path, parameters=parameters)
    resumed.start_run(summary)
    failed = prediction.model_copy(
        update={
            "predicted_personal": None,
            "outcome": "FAILED",
            "status": "FAILED",
            "error_message": "RuntimeError",
        }
    )
    resumed.record_case(case, result, failed)
    resumed.finalize(summary)

    with resumed.paths.results.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["benchmark_id"] == "1"
    assert rows[0]["status"] == "FAILED"
