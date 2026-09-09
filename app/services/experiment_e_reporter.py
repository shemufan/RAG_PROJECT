"""Run-isolated CSV and JSON diagnostics for Experiment E."""

import csv
import json
from pathlib import Path
from typing import NamedTuple

from app.schemas.classification import SemanticBridgeTrace


class ExperimentEOutputPaths(NamedTuple):
    results: Path
    summary: Path
    semantic_debug: Path


class ExperimentEReporter:
    """Collect per-case traces and emit the three requested artifacts."""

    def __init__(self, output_root: str | Path, *, parameters: dict) -> None:
        self.output_root = Path(output_root)
        self.parameters = parameters
        self._summary = None
        self._result_rows: list[dict] = []
        self._debug_rows: list[dict] = []
        self.paths: ExperimentEOutputPaths | None = None

    def start_run(self, summary) -> None:
        self._summary = summary
        directory = self.output_root / "experiment_E" / str(summary.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = ExperimentEOutputPaths(
            results=directory / "experiment_E_results.csv",
            summary=directory / "experiment_E_summary.json",
            semantic_debug=directory / "semantic_retrieval_debug.csv",
        )
        self._result_rows = _read_csv(self.paths.results)
        self._debug_rows = _read_csv(self.paths.semantic_debug)

    def record_case(self, case, result, prediction) -> None:
        trace = (
            result.experiment_trace
            if result is not None and result.experiment_trace is not None
            else SemanticBridgeTrace(failed_stage="classification")
        )
        semantic_rows = [
            {
                "semantic_type": item.card.semantic_type,
                "aliases": item.card.aliases,
                "semantic_category": item.card.semantic_category,
                "regulation_keywords": item.card.regulation_keywords,
                "description": item.card.description,
                "raw_score": item.raw_score,
            }
            for item in trace.semantic_retrieval
        ]
        regulation_rows = [
            {
                "chunk": item.evidence.content,
                "raw_score": item.raw_score,
                "source": item.evidence.source,
                "article": item.evidence.article,
                "chunk_id": item.evidence.chunk_id,
            }
            for item in trace.regulation_retrieval
        ]
        expected = case.expected_personal
        predicted = prediction.predicted_personal
        correct = (
            expected == predicted
            if expected is not None and predicted is not None
            else None
        )
        result_row = {
                "benchmark_id": case.case_index,
                "field_name": case.field_profile.field_name,
                "sample_values": _json(case.field_profile.sample_values),
                "profiling": _json(trace.profiling.model_dump()),
                "semantic_query": trace.semantic_query,
                "semantic_retrieval": _json(semantic_rows),
                "selected_semantic_type": trace.selected_semantic_type,
                "regulation_query": trace.regulation_query,
                "regulation_retrieval": _json(regulation_rows),
                "prediction": predicted,
                "ground_truth": expected,
                "correct": correct,
                "outcome": prediction.outcome,
                "status": prediction.status,
                "failed_stage": trace.failed_stage,
                "error_message": prediction.error_message,
                "category": prediction.category,
                "subcategory": prediction.subcategory,
                "level": prediction.level,
                "confidence": prediction.confidence,
                "reason": prediction.reason,
                "need_review": prediction.need_review,
            }
        debug_row = self._semantic_debug_row(case, trace)
        self._result_rows = [
            row
            for row in self._result_rows
            if str(row.get("benchmark_id")) != str(case.case_index)
        ]
        self._debug_rows = [
            row
            for row in self._debug_rows
            if str(row.get("benchmark_id")) != str(case.case_index)
        ]
        self._result_rows.append(result_row)
        self._debug_rows.append(debug_row)
        assert self.paths is not None
        _write_csv_atomic(self.paths.results, self._result_rows)
        _write_csv_atomic(self.paths.semantic_debug, self._debug_rows)

    def finalize(self, summary) -> ExperimentEOutputPaths:
        if self.paths is None:
            self.start_run(summary)
        assert self.paths is not None
        _write_csv_atomic(self.paths.results, self._result_rows)
        _write_csv_atomic(self.paths.semantic_debug, self._debug_rows)
        payload = {
            "experiment": "E",
            "run_id": str(summary.run_id),
            "parameters": self.parameters,
            "model_name": summary.model_name,
            "knowledge_base_version": summary.knowledge_base_version,
            "metrics": summary.model_dump(mode="json"),
            "outputs": {
                name: str(path)
                for name, path in zip(self.paths._fields, self.paths)
            },
        }
        _write_text_atomic(
            self.paths.summary,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return self.paths

    @staticmethod
    def _semantic_debug_row(case, trace: SemanticBridgeTrace) -> dict:
        row = {
            "benchmark_id": case.case_index,
            "field_name": case.field_profile.field_name,
            "samples": _json(case.field_profile.sample_values),
            "profiling": _json(trace.profiling.model_dump()),
            "selected_semantic_type": trace.selected_semantic_type,
            "top1_top2_score_gap": trace.top1_top2_score_gap,
            "failed_stage": trace.failed_stage,
        }
        for index in range(3):
            result = (
                trace.semantic_retrieval[index]
                if index < len(trace.semantic_retrieval)
                else None
            )
            number = index + 1
            row[f"top{number}_semantic_type"] = (
                result.card.semantic_type if result else None
            )
            row[f"top{number}_raw_score"] = result.raw_score if result else None
        return row


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _fieldnames(rows: list[dict]) -> list[str]:
    return list(rows[0]) if rows else []


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8-sig")
    temporary.replace(path)
