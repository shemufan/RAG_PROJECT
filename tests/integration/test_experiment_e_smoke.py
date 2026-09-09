from datetime import datetime, timezone
from uuid import UUID

from app.schemas.classification import (
    ClassificationOutput,
    Evidence,
    RegulationRetrievalResult,
)
from app.schemas.csv_input import CSVFieldCase, CSVInputBatch
from app.schemas.field import FieldProfile
from app.schemas.semantic import SemanticCard, SemanticRetrievalResult
from app.services.csv_pipeline import CSVClassificationPipeline
from app.services.experiment_e_reporter import ExperimentEReporter
from app.services.semantic_bridge_service import SemanticBridgeClassificationService


def test_experiment_e_end_to_end_with_injected_dependencies(tmp_path):
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    field = FieldProfile(
        field_name="contact_value",
        sample_values=["13812345678", "15987654321"],
    )
    card = SemanticCard(
        semantic_type="手机号码",
        aliases=["手机号", "phone"],
        common_field_names=["phone"],
        value_features=["通常为11位数字"],
        description="用于联系自然人的电话号码",
        semantic_category=["联系方式", "个人信息"],
        regulation_keywords=["手机号码", "联系方式"],
    )

    class Profiler:
        def basic_statistics(self, values):
            return ["字符串长度约11位", "数字字符占比100%"]

    class SemanticStore:
        def search(self, query, k):
            assert "字段客观特征" in query
            return [SemanticRetrievalResult(card=card, raw_score=0.82)]

    class RegulationStore:
        def search_raw(self, query, k):
            assert "手机号码" in query
            assert "13812345678" not in query
            return [
                RegulationRetrievalResult(
                    evidence=Evidence(
                        content="电话号码属于个人信息",
                        source="rules.md",
                        score=0.73,
                    ),
                    raw_score=0.73,
                )
            ]

    class LLM:
        def classify(self, prompt):
            assert "Semantic Knowledge" in prompt[1].content
            return ClassificationOutput(
                is_personal=True,
                category="个人基本资料",
                subcategory="手机号码",
                level="L3",
                confidence=0.92,
                reason="字段语义和法规依据一致",
                need_review=False,
            )

    class Repository:
        def __init__(self):
            self.predictions = []

        def create_run(self, summary):
            self.summary = summary

        def save_prediction(self, prediction):
            self.predictions.append(prediction)

        def load_metric_inputs(self, run_id_value):
            return [row.metric_input() for row in self.predictions]

        def update_run(self, summary):
            self.summary = summary

    reporter = ExperimentEReporter(
        tmp_path,
        parameters={"semantic_top_k": 3, "regulation_top_k": 3},
    )
    pipeline = CSVClassificationPipeline(
        Repository(),
        SemanticBridgeClassificationService(
            SemanticStore(),
            RegulationStore(),
            LLM(),
            value_profiler=Profiler(),
        ),
        model_name="same-as-b",
        knowledge_base_version="v1",
        clock=lambda: now,
        run_id_factory=lambda: run_id,
        experiment_observer=reporter,
    )
    batch = CSVInputBatch(
        source_name="benchmark.csv",
        source_fingerprint="a" * 64,
        input_mode="catalog",
        cases=[CSVFieldCase(case_index=1, field_profile=field, expected_personal=True)],
    )

    summary = pipeline.run(batch)

    assert summary.tp == 1
    assert summary.f1_score == 1.0
    assert reporter.paths is not None
    assert all(path.is_file() for path in reporter.paths)

