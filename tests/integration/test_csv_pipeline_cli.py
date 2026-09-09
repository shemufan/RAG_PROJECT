import csv
from types import SimpleNamespace

import pytest

import scripts.run_csv_pipeline as csv_cli
from scripts.run_csv_pipeline import apply_limit, load_csv_batch, load_labels, parse_args


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_cli_auto_loads_tabular_and_catalog_inputs(tmp_path):
    tabular = tmp_path / "business.csv"
    catalog = tmp_path / "catalog.csv"
    write_csv(tabular, [["email", "price"], ["a***@x.test", "99"]])
    write_csv(catalog, [["字段名", "样本1"], ["email", "a***@x.test"]])

    tabular_batch = load_csv_batch(parse_args(["--input", str(tabular)]))
    catalog_batch = load_csv_batch(parse_args(["--input", str(catalog)]))

    assert tabular_batch.input_mode == "tabular"
    assert catalog_batch.input_mode == "catalog"


def test_cli_accepts_resume_and_rejects_invalid_combinations(tmp_path):
    path = tmp_path / "business.csv"
    write_csv(path, [["email"], ["a***@x.test"]])
    run_id = "12345678-1234-5678-1234-567812345678"

    resumed = parse_args(
        ["--resume-run", run_id, "--input", str(path), "--retry-failed"]
    )
    assert str(resumed.resume_run) == run_id

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--retry-failed"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--input-mode",
                "tabular",
                "--field-name-column",
                "name",
            ]
        )


def test_cli_loads_embedded_label_column(tmp_path):
    source = tmp_path / "catalog.csv"
    write_csv(
        source,
        [
            ["字段名", "样本1", "expected_personal"],
            ["email", "a***@x.test", "true"],
            ["price", "99", "false"],
        ],
    )

    args = parse_args(["--input", str(source), "--label-column", "expected_personal"])
    batch = load_csv_batch(args)
    labels = load_labels(batch, args)

    assert batch.input_mode == "catalog"
    assert [case.expected_personal for case in labels.cases] == [True, False]


def test_cli_rejects_label_column_with_separate_labels(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1", "expected_personal"], ["email", "a", "true"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--label-column",
                "expected_personal",
                "--labels",
                "labels.csv",
            ]
        )


def test_cli_rejects_label_column_with_tabular_mode(tmp_path):
    path = tmp_path / "business.csv"
    write_csv(path, [["email", "price"], ["a", "1"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--label-column",
                "expected_personal",
                "--input-mode",
                "tabular",
            ]
        )


def test_cli_limit_truncates_batch_and_labels(tmp_path):
    source = tmp_path / "catalog.csv"
    write_csv(
        source,
        [
            ["字段名", "样本1", "expected_personal"],
            ["email", "a", "true"],
            ["price", "99", "false"],
            ["ip", "1.2.3.4", "true"],
        ],
    )

    args = parse_args(
        ["--input", str(source), "--label-column", "expected_personal", "--limit", "2"]
    )
    batch = load_csv_batch(args)
    labels = load_labels(batch, args)
    limited_batch, limited_labels = apply_limit(batch, labels, args.limit)

    assert [case.field_profile.field_name for case in limited_batch.cases] == [
        "email",
        "price",
    ]
    assert [case.expected_personal for case in limited_labels.cases] == [True, False]
    assert limited_labels.labeled_cases == 2
    assert limited_labels.unlabeled_cases == 0


def test_cli_rejects_limit_with_resume(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a"]])
    run_id = "12345678-1234-5678-1234-567812345678"

    with pytest.raises(SystemExit):
        parse_args(
            ["--input", str(path), "--limit", "5", "--resume-run", run_id]
        )


@pytest.mark.parametrize("strategy", ["legacy", "clean", "profile"])
def test_cli_accepts_query_strategy(tmp_path, strategy: str):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(["--input", str(path), "--query-strategy", strategy])

    assert args.query_strategy == strategy


def test_cli_defaults_query_strategy_to_profile(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).query_strategy == "profile"


def test_cli_rejects_unknown_query_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--query-strategy", "other"])


@pytest.mark.parametrize("mode", ["c", "c1", "c2"])
def test_cli_accepts_profile_query_mode(tmp_path, mode: str):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(
        [
            "--input",
            str(path),
            "--query-strategy",
            "profile",
            "--query-mode",
            mode,
        ]
    )

    assert args.query_mode == mode


def test_cli_defaults_profile_query_mode_to_full_c(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).query_mode == "c"


def test_cli_rejects_unknown_profile_query_mode(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--query-mode", "other"])


def test_cli_rejects_profile_submode_for_non_profile_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--query-strategy",
                "clean",
                "--query-mode",
                "c1",
            ]
        )


@pytest.mark.parametrize("mode", ["rule", "llm"])
def test_cli_accepts_profiling_mode(tmp_path, mode: str):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(["--input", str(path), "--profiling-mode", mode])

    assert args.profiling_mode == mode


def test_cli_defaults_profiling_mode_to_rule(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).profiling_mode == "rule"


def test_cli_rejects_unknown_profiling_mode(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(["--input", str(path), "--profiling-mode", "other"])


def test_cli_rejects_llm_profiling_for_non_profile_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--query-strategy",
                "clean",
                "--profiling-mode",
                "llm",
            ]
        )


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--use-rag", True), ("--no-use-rag", False)],
)
def test_cli_accepts_rag_switch(tmp_path, flag: str, expected: bool):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(
        ["--input", str(path), "--query-strategy", "clean", flag]
    )

    assert args.use_rag is expected


def test_cli_enables_rag_by_default(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    assert parse_args(["--input", str(path)]).use_rag is True


@pytest.mark.parametrize(
    ("experiment", "strategy", "profiling_mode", "use_rag"),
    [
        ("A", "legacy", "rule", True),
        ("B", "clean", "rule", True),
        ("C", "profile", "rule", True),
        ("D", "clean", "rule", False),
        ("E", "clean", "rule", True),
    ],
)
def test_cli_resolves_experiment_presets(
    tmp_path,
    experiment,
    strategy,
    profiling_mode,
    use_rag,
):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    args = parse_args(["--input", str(path), "--experiment", experiment])

    assert args.experiment == experiment
    assert args.query_strategy == strategy
    assert args.profiling_mode == profiling_mode
    assert args.use_rag is use_rag
    assert args.semantic_top_k == 3
    assert args.regulation_top_k == 3


def test_cli_rejects_flags_that_conflict_with_experiment_e(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--experiment",
                "E",
                "--no-use-rag",
            ]
        )


def test_cli_rejects_no_rag_for_non_clean_strategy(tmp_path):
    path = tmp_path / "catalog.csv"
    write_csv(path, [["字段名", "样本1"], ["email", "a***@x.test"]])

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                str(path),
                "--query-strategy",
                "profile",
                "--no-use-rag",
            ]
        )


def test_build_pipeline_passes_llm_profiler_to_classifier(monkeypatch):
    captured = {}
    profiler = object()

    class FakeVectorStore:
        def __init__(self, embedding_service, *, settings):
            pass

        def count(self) -> int:
            return 1

    class CapturingClassifier:
        def __init__(
            self,
            vector_store,
            llm_service,
            *,
            query_strategy,
            profile_query_mode,
            value_profiler,
            use_rag,
        ):
            captured["query_strategy"] = query_strategy
            captured["profile_query_mode"] = profile_query_mode
            captured["value_profiler"] = value_profiler
            captured["use_rag"] = use_rag

    monkeypatch.setattr(csv_cli, "EmbeddingService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(csv_cli, "LLMService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "LLMValueProfiler", lambda **kwargs: profiler)
    monkeypatch.setattr(csv_cli, "FieldClassificationService", CapturingClassifier)
    monkeypatch.setattr(csv_cli, "BenchmarkTargetRepository", lambda url: object())
    settings = SimpleNamespace(
        target_database_url="sqlite://",
        embedding_model_path="model",
        deepseek_model="llm",
        knowledge_base_version="kb",
    )

    csv_cli.build_pipeline(
        settings,
        query_strategy="profile",
        profile_query_mode="c2",
        profiling_mode="llm",
    )

    assert captured == {
        "query_strategy": "profile",
        "profile_query_mode": "c2",
        "value_profiler": profiler,
        "use_rag": True,
    }


def test_build_pipeline_keeps_rule_profiler_as_default(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def __init__(self, embedding_service, *, settings):
            pass

        def count(self) -> int:
            return 1

    class CapturingClassifier:
        def __init__(self, vector_store, llm_service, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(csv_cli, "EmbeddingService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(csv_cli, "LLMService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "FieldClassificationService", CapturingClassifier)
    monkeypatch.setattr(csv_cli, "BenchmarkTargetRepository", lambda url: object())
    settings = SimpleNamespace(
        target_database_url="sqlite://",
        embedding_model_path="model",
        deepseek_model="llm",
        knowledge_base_version="kb",
    )

    csv_cli.build_pipeline(settings)

    assert captured == {
        "query_strategy": "profile",
        "profile_query_mode": "c",
        "value_profiler": None,
        "use_rag": True,
    }


def test_build_pipeline_passes_no_rag_to_classifier(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def __init__(self, embedding_service, *, settings):
            pass

        def count(self) -> int:
            return 1

    class CapturingClassifier:
        def __init__(self, vector_store, llm_service, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(csv_cli, "EmbeddingService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(csv_cli, "LLMService", lambda **kwargs: object())
    monkeypatch.setattr(csv_cli, "FieldClassificationService", CapturingClassifier)
    monkeypatch.setattr(csv_cli, "BenchmarkTargetRepository", lambda url: object())
    settings = SimpleNamespace(
        target_database_url="sqlite://",
        embedding_model_path="model",
        deepseek_model="llm",
        knowledge_base_version="kb",
    )

    csv_cli.build_pipeline(
        settings,
        query_strategy="clean",
        use_rag=False,
    )

    assert captured == {
        "query_strategy": "clean",
        "profile_query_mode": "c",
        "value_profiler": None,
        "use_rag": False,
    }


def test_build_pipeline_wires_independent_stores_for_experiment_e(
    monkeypatch,
    tmp_path,
):
    captured = {"embeddings": []}
    embedding = object()

    class FakeRegulationStore:
        def __init__(self, embedding_service, *, settings):
            captured["embeddings"].append(embedding_service)

        def count(self):
            return 2

    class FakeSemanticStore:
        def __init__(self, embedding_service, *, settings):
            captured["embeddings"].append(embedding_service)

        def count(self):
            return 33

    class FakeBridgeService:
        def __init__(self, semantic_store, regulation_store, llm, **kwargs):
            captured["bridge"] = (semantic_store, regulation_store, llm, kwargs)
            captured["bridge_instance"] = self

    class FakeReporter:
        def __init__(self, output_root, *, parameters):
            captured["reporter"] = (output_root, parameters)
            captured["reporter_instance"] = self

    monkeypatch.setattr(csv_cli, "EmbeddingService", lambda **kwargs: embedding)
    monkeypatch.setattr(csv_cli, "VectorStore", FakeRegulationStore)
    monkeypatch.setattr(csv_cli, "SemanticVectorStore", FakeSemanticStore)
    monkeypatch.setattr(csv_cli, "SemanticBridgeClassificationService", FakeBridgeService)
    monkeypatch.setattr(csv_cli, "ExperimentEReporter", FakeReporter)
    monkeypatch.setattr(csv_cli, "LLMService", lambda **kwargs: "llm")
    monkeypatch.setattr(csv_cli, "BenchmarkTargetRepository", lambda url: "repository")
    settings = SimpleNamespace(
        target_database_url="sqlite://",
        embedding_model_path="model",
        deepseek_model="llm-name",
        knowledge_base_version="kb",
    )

    pipeline = csv_cli.build_pipeline(
        settings,
        experiment="E",
        semantic_top_k=5,
        regulation_top_k=3,
        output_root=tmp_path,
    )

    assert captured["embeddings"] == [embedding, embedding]
    assert captured["bridge"][3] == {
        "semantic_top_k": 5,
        "regulation_top_k": 3,
    }
    assert captured["reporter"][0] == tmp_path
    assert captured["reporter"][1]["use_extra_semantic_llm"] is False
    assert pipeline.classification_service is captured["bridge_instance"]
    assert pipeline.experiment_observer is captured["reporter_instance"]
