import pytest

from scripts.rebuild_knowledge_base import create_parser


def test_quality_cli_accepts_check_and_candidate_options():
    args = create_parser().parse_args(
        [
            "--check-only",
            "--candidate-collection",
            "candidate-v2",
            "--approve-review",
            "a" * 64,
        ]
    )

    assert args.check_only is True
    assert args.candidate_collection == "candidate-v2"
    assert args.approve_review == ["a" * 64]


def test_quality_cli_rejects_unknown_option():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["--unsafe-overwrite-current"])
