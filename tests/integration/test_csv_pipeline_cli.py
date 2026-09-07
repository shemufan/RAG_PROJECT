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


def test_build_pipeline_passes_query_strategy_to_classifier(monkeypatch):
    captured = {}

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
        ):
            captured["query_strategy"] = query_strategy
            captured["profile_query_mode"] = profile_query_mode

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
        query_strategy="profile",
        profile_query_mode="c2",
    )

    assert captured == {
        "query_strategy": "profile",
        "profile_query_mode": "c2",
    }
