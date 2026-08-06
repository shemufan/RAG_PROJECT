import csv

import pytest

from scripts.run_benchmark import load_benchmark_input, parse_args


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_benchmark_cli_loads_two_catalog_files_with_independent_limits(tmp_path):
    personal = tmp_path / "personal.csv"
    non_personal = tmp_path / "non_personal.csv"
    write_csv(personal, [["field_name", "sample1"], ["email", "a"], ["phone", "b"]])
    write_csv(
        non_personal,
        [["field_name", "sample1"], ["price", "1"], ["created_at", "2"]],
    )
    args = parse_args(
        [
            "--personal",
            str(personal),
            "--non-personal",
            str(non_personal),
            "--batch",
            "teacher",
            "--personal-limit",
            "1",
            "--non-personal-limit",
            "1",
        ]
    )

    prepared = load_benchmark_input(args)

    assert [case.field_profile.field_name for case in prepared.batch.cases] == [
        "email",
        "price",
    ]
    assert prepared.batch.personal_limit == 1
    assert prepared.batch.non_personal_limit == 1


def test_benchmark_cli_requires_both_files_and_valid_resume_options():
    run_id = "12345678-1234-5678-1234-567812345678"
    common = ["--personal", "personal.csv", "--non-personal", "other.csv"]

    resumed = parse_args([*common, "--resume-run", run_id, "--retry-failed"])
    assert str(resumed.resume_run) == run_id

    with pytest.raises(SystemExit):
        parse_args(["--personal", "personal.csv", "--batch", "teacher"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--batch", "teacher", "--retry-failed"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--resume-run", run_id, "--personal-limit", "1"])
