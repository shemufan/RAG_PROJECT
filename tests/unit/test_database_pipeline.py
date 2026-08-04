from app.schemas.classification import ClassificationResult, Evidence
from app.schemas.field import FieldProfile
from app.schemas.pipeline import PipelineRequest
from app.services.database_pipeline import DatabaseClassificationPipeline


def make_profile(field_name: str) -> FieldProfile:
    return FieldProfile(
        source_system="mysql",
        database_name="enterprise_source",
        table_name="employee",
        field_name=field_name,
        field_comment=f"{field_name} comment",
        data_type="varchar(64)",
        is_nullable=True,
        business_domain="hr",
    )


def make_result(field_name: str, *, need_review: bool = False) -> ClassificationResult:
    return ClassificationResult(
        field_name=field_name,
        is_personal=True,
        category="个人信息",
        subcategory="身份标识",
        level="L3",
        confidence=0.85,
        reason="法规依据",
        evidence=[
            Evidence(source="first.txt", content="first", score=0.9),
            Evidence(source="second.txt", content="second", score=0.8),
        ],
        need_review=need_review,
        decision_path="rag_llm",
    )


def unknown_result(field_name: str) -> ClassificationResult:
    return ClassificationResult(
        field_name=field_name,
        is_personal=None,
        category="未知",
        level="UNKNOWN",
        confidence=0.0,
        reason="分类处理失败，请进行人工复核。",
        evidence=[],
        need_review=True,
        decision_path="rag_llm_error",
    )


class FakeSourceRepository:
    database_name = "enterprise_source"

    def __init__(self, profiles=None, error=None):
        self.profiles = profiles or []
        self.error = error
        self.requests = []

    def scan_fields(self, sample_limit=3, table_names=None):
        self.requests.append((sample_limit, table_names))
        if self.error:
            raise self.error
        return self.profiles


class FakeTargetRepository:
    def __init__(self):
        self.created_runs = []
        self.updated_runs = []
        self.upserted_assets = []
        self.saved_records = []

    def create_run(self, summary, model_name, knowledge_version):
        self.created_runs.append((summary, model_name, knowledge_version))

    def update_run(self, summary, error_message=None):
        self.updated_runs.append((summary, error_message))

    def upsert_field_asset(self, field_id, profile, seen_at):
        self.upserted_assets.append((field_id, profile, seen_at))

    def save_classification_record(self, record):
        self.saved_records.append(record)
        return len(self.saved_records)


class FakeClassifier:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.fields = []

    def classify_field(self, profile):
        self.fields.append(profile.field_name)
        outcome = self.outcomes[profile.field_name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_pipeline(profiles, outcomes, *, source_error=None):
    source = FakeSourceRepository(profiles, error=source_error)
    target = FakeTargetRepository()
    classifier = FakeClassifier(outcomes)
    pipeline = DatabaseClassificationPipeline(
        source,
        target,
        classifier,
        model_name="deepseek-chat",
        knowledge_base_version="v1",
    )
    return pipeline, source, target, classifier


def test_pipeline_success_uses_new_run_and_stable_field_ids():
    profile = make_profile("employee_name")
    pipeline, source, target, _ = make_pipeline(
        [profile],
        {profile.field_name: make_result(profile.field_name)},
    )

    first = pipeline.run(PipelineRequest())
    second = pipeline.run(PipelineRequest())

    assert first.run_id != second.run_id
    assert target.saved_records[0].field_id == target.saved_records[1].field_id
    assert first.status == "SUCCESS"
    assert first.total_fields == first.success_fields == 1
    assert first.failed_fields == 0
    assert source.requests == [(3, None), (3, None)]
    assert [item.source for item in target.saved_records[0].evidence] == [
        "first.txt",
        "second.txt",
    ]


def test_pipeline_counts_review_without_failure():
    profile = make_profile("home_address")
    pipeline, _, _, _ = make_pipeline(
        [profile],
        {profile.field_name: make_result(profile.field_name, need_review=True)},
    )

    summary = pipeline.run(PipelineRequest())

    assert summary.status == "SUCCESS"
    assert summary.review_fields == 1
    assert summary.failed_fields == 0


def test_pipeline_continues_after_unknown_and_does_not_save_failed_result():
    failed = make_profile("id_card_no")
    succeeded = make_profile("department")
    pipeline, _, target, classifier = make_pipeline(
        [failed, succeeded],
        {
            failed.field_name: unknown_result(failed.field_name),
            succeeded.field_name: make_result(succeeded.field_name),
        },
    )

    summary = pipeline.run(PipelineRequest(continue_on_error=True))

    assert summary.status == "PARTIAL_FAILED"
    assert summary.success_fields == 1
    assert summary.failed_fields == 1
    assert classifier.fields == ["id_card_no", "department"]
    assert len(target.saved_records) == 1
    assert len(target.upserted_assets) == 2


def test_pipeline_all_unknown_is_failed():
    profiles = [make_profile("id_card_no"), make_profile("phone")]
    pipeline, _, target, _ = make_pipeline(
        profiles,
        {profile.field_name: unknown_result(profile.field_name) for profile in profiles},
    )

    summary = pipeline.run(PipelineRequest())

    assert summary.status == "FAILED"
    assert summary.success_fields == 0
    assert summary.failed_fields == summary.total_fields == 2
    assert target.saved_records == []


def test_pipeline_stops_after_first_failure_when_requested():
    failed = make_profile("id_card_no")
    unprocessed = make_profile("department")
    pipeline, _, target, classifier = make_pipeline(
        [failed, unprocessed],
        {
            failed.field_name: RuntimeError("provider unavailable"),
            unprocessed.field_name: make_result(unprocessed.field_name),
        },
    )

    summary = pipeline.run(PipelineRequest(continue_on_error=False))

    assert summary.status == "FAILED"
    assert summary.total_fields == 2
    assert summary.failed_fields == 1
    assert classifier.fields == ["id_card_no"]
    assert len(target.upserted_assets) == 1


def test_pipeline_scan_failure_finalizes_failed_run():
    pipeline, _, target, _ = make_pipeline(
        [],
        {},
        source_error=RuntimeError("metadata unavailable"),
    )

    summary = pipeline.run(PipelineRequest(table_names=["employee"]))

    assert summary.status == "FAILED"
    assert summary.total_fields == 0
    assert summary.finished_at is not None
    assert target.created_runs[0][0].status == "RUNNING"
    assert target.updated_runs[-1][0].status == "FAILED"
    assert "metadata unavailable" in target.updated_runs[-1][1]
