"""Pure personal-information benchmark metric calculation."""

from collections.abc import Iterable

from app.schemas.benchmark import BenchmarkMetricInput, BenchmarkMetrics


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    predictions: Iterable[BenchmarkMetricInput],
) -> BenchmarkMetrics:
    """Calculate model and end-to-end metrics from case-level predictions."""
    rows = list(predictions)
    successful = [row for row in rows if row.predicted_personal is not None]
    tp = sum(row.expected_personal and row.predicted_personal is True for row in successful)
    fn = sum(row.expected_personal and row.predicted_personal is False for row in successful)
    fp = sum(not row.expected_personal and row.predicted_personal is True for row in successful)
    tn = sum(not row.expected_personal and row.predicted_personal is False for row in successful)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return BenchmarkMetrics(
        total_cases=len(rows),
        success_cases=len(successful),
        failed_cases=len(rows) - len(successful),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_score=precision,
        recall_score=recall,
        f1_score=_safe_divide(2 * precision * recall, precision + recall),
        accuracy_score=_safe_divide(tp + tn, len(successful)),
        coverage_score=_safe_divide(len(successful), len(rows)),
        effective_recall_score=_safe_divide(
            tp,
            sum(row.expected_personal for row in rows),
        ),
    )
