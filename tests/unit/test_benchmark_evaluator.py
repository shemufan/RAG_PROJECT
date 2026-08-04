from app.schemas.benchmark import BenchmarkMetricInput
from app.services.benchmark_evaluator import evaluate_predictions


def test_evaluator_calculates_confusion_matrix_and_scores():
    metrics = evaluate_predictions(
        [
            BenchmarkMetricInput(expected_personal=True, predicted_personal=True),
            BenchmarkMetricInput(expected_personal=True, predicted_personal=False),
            BenchmarkMetricInput(expected_personal=False, predicted_personal=True),
            BenchmarkMetricInput(expected_personal=False, predicted_personal=False),
        ]
    )

    assert (metrics.tp, metrics.fn, metrics.fp, metrics.tn) == (1, 1, 1, 1)
    assert metrics.precision_score == 0.5
    assert metrics.recall_score == 0.5
    assert metrics.f1_score == 0.5
    assert metrics.accuracy_score == 0.5
    assert metrics.coverage_score == 1.0
    assert metrics.effective_recall_score == 0.5


def test_evaluator_counts_failure_in_coverage_and_effective_recall():
    metrics = evaluate_predictions(
        [
            BenchmarkMetricInput(expected_personal=True, predicted_personal=True),
            BenchmarkMetricInput(expected_personal=True, predicted_personal=None),
            BenchmarkMetricInput(expected_personal=False, predicted_personal=None),
        ]
    )

    assert metrics.success_cases == 1
    assert metrics.failed_cases == 2
    assert metrics.coverage_score == 1 / 3
    assert metrics.recall_score == 1.0
    assert metrics.effective_recall_score == 0.5


def test_evaluator_returns_zero_for_empty_or_zero_denominators():
    empty = evaluate_predictions([])
    negatives = evaluate_predictions(
        [BenchmarkMetricInput(expected_personal=False, predicted_personal=False)]
    )

    assert empty.precision_score == empty.recall_score == empty.f1_score == 0.0
    assert negatives.precision_score == negatives.recall_score == 0.0
