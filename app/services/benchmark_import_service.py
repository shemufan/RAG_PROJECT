"""Decode and validate labeled benchmark CSV files without model calls."""

import csv
import io
import re
from pathlib import Path

from app.schemas.benchmark import BenchmarkImportRow, DatasetLabel

SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
SAMPLE_HEADER = re.compile(r"^样本(\d+)$")


class BenchmarkImportError(ValueError):
    """Raised when a benchmark CSV cannot be decoded or validated."""


def _decode_csv(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BenchmarkImportError(f"cannot read CSV: {path.name}") from exc
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise BenchmarkImportError(f"unsupported CSV encoding: {path.name}")


def parse_benchmark_csv(
    path: str | Path,
    *,
    source_dataset: DatasetLabel,
) -> list[BenchmarkImportRow]:
    """Return validated rows with labels assigned only from the caller."""
    csv_path = Path(path)
    reader = csv.DictReader(io.StringIO(_decode_csv(csv_path), newline=""))
    headers = reader.fieldnames or []
    if not headers or headers[0].strip() != "字段名":
        raise BenchmarkImportError(f"{csv_path.name}: first column must be 字段名")
    sample_headers = sorted(
        (
            (int(match.group(1)), header)
            for header in headers[1:]
            if (match := SAMPLE_HEADER.fullmatch(header.strip()))
        ),
        key=lambda item: item[0],
    )
    if not sample_headers:
        raise BenchmarkImportError(f"{csv_path.name}: no 样本N columns found")

    expected = source_dataset == "personal"
    parsed = []
    for source_row_number, raw in enumerate(reader, start=2):
        field_name = (raw.get(headers[0]) or "").strip()
        if not field_name:
            raise BenchmarkImportError(
                f"{csv_path.name}: row {source_row_number} has empty 字段名"
            )
        samples = []
        for _, header in sample_headers:
            value = (raw.get(header) or "").strip()
            if value:
                samples.append(value[:50])
            if len(samples) == 5:
                break
        parsed.append(
            BenchmarkImportRow(
                source_dataset=source_dataset,
                source_row_number=source_row_number,
                field_name=field_name,
                sample_values=samples,
                expected_personal=expected,
            )
        )
    return parsed
