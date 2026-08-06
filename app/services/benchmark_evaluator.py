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
    labeled = [row for row in rows if row.expected_personal is not None]
    successful = [row for row in rows if row.predicted_personal is not None]
    successful_labeled = [
        row for row in labeled if row.predicted_personal is not None
    ]
    tp = sum(
        row.expected_personal is True and row.predicted_personal is True
        for row in successful_labeled
    )
    fn = sum(
        row.expected_personal is True and row.predicted_personal is False
        for row in successful_labeled
    )
    fp = sum(
        row.expected_personal is False and row.predicted_personal is True
        for row in successful_labeled
    )
    tn = sum(
        row.expected_personal is False and row.predicted_personal is False
        for row in successful_labeled
    )
    if not labeled:
        return BenchmarkMetrics(
            total_cases=len(rows),
            success_cases=len(successful),
            failed_cases=len(rows) - len(successful),
            labeled_cases=0,
            unlabeled_cases=len(rows),
            precision_score=None,
            recall_score=None,
            f1_score=None,
            accuracy_score=None,
            coverage_score=_safe_divide(len(successful), len(rows)),
            effective_recall_score=None,
        )
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return BenchmarkMetrics(
        total_cases=len(rows),
        success_cases=len(successful),
        failed_cases=len(rows) - len(successful),
        labeled_cases=len(labeled),
        unlabeled_cases=len(rows) - len(labeled),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_score=precision,
        recall_score=recall,
        f1_score=_safe_divide(2 * precision * recall, precision + recall),
        accuracy_score=_safe_divide(tp + tn, len(successful_labeled)),
        coverage_score=_safe_divide(len(successful), len(rows)),
        effective_recall_score=_safe_divide(
            tp,
            sum(row.expected_personal is True for row in labeled),
        ),
    )
