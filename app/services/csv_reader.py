"""Validated, bounded, streaming reads for CSV input adapters."""

import codecs
import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from app.schemas.csv_input import CSVInspection, CSVReadLimits

SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
CHUNK_SIZE = 64 * 1024


class CSVInputError(ValueError):
    """Raised when an input CSV cannot be safely normalized."""


class CSVReader:
    def __init__(self, limits: CSVReadLimits | None = None):
        self.limits = limits or CSVReadLimits()

    def inspect(self, path: str | Path) -> CSVInspection:
        csv_path = Path(path)
        try:
            size = csv_path.stat().st_size
        except OSError as exc:
            raise CSVInputError(f"cannot read CSV: {csv_path.name}") from exc
        if size == 0:
            raise CSVInputError("empty CSV")
        if size > self.limits.max_bytes:
            raise CSVInputError("CSV size exceeds configured limit")

        encoding = self._detect_encoding(csv_path)
        fingerprint = self._sha256(csv_path)
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                raw_headers = next(reader, None)
                if not raw_headers:
                    raise CSVInputError("empty CSV")
                headers = [header.strip() for header in raw_headers]
                self._validate_headers(headers)
                row_count = 0
                for row_number, row in enumerate(reader, start=2):
                    self._validate_row_width(row, headers, row_number)
                    row_count += 1
                    if row_count > self.limits.max_rows:
                        raise CSVInputError("CSV rows exceed configured limit")
        except (OSError, UnicodeError, csv.Error) as exc:
            raise CSVInputError(f"invalid CSV: {csv_path.name}") from exc
        return CSVInspection(
            source_name=csv_path.name,
            source_fingerprint=fingerprint,
            encoding=encoding,
            headers=headers,
            row_count=row_count,
        )

    def iter_rows(
        self,
        path: str | Path,
        inspection: CSVInspection,
    ) -> Iterator[tuple[int, dict[str, str]]]:
        csv_path = Path(path)
        try:
            with csv_path.open(
                "r",
                encoding=inspection.encoding,
                newline="",
            ) as handle:
                reader = csv.reader(handle)
                next(reader)
                for row_number, row in enumerate(reader, start=2):
                    self._validate_row_width(row, inspection.headers, row_number)
                    yield row_number, dict(zip(inspection.headers, row, strict=True))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise CSVInputError(f"invalid CSV: {csv_path.name}") from exc

    def _validate_headers(self, headers: list[str]) -> None:
        if len(headers) > self.limits.max_columns:
            raise CSVInputError("CSV columns exceed configured limit")
        if any(not header for header in headers):
            raise CSVInputError("blank header is not allowed")
        if len(set(headers)) != len(headers):
            raise CSVInputError("duplicate header is not allowed")

    @staticmethod
    def _validate_row_width(row: list[str], headers: list[str], row_number: int) -> None:
        if len(row) != len(headers):
            raise CSVInputError(f"row {row_number} has an invalid column count")

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        for encoding in SUPPORTED_ENCODINGS:
            decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
            try:
                with path.open("rb") as handle:
                    while chunk := handle.read(CHUNK_SIZE):
                        decoder.decode(chunk)
                    decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                continue
            return encoding
        raise CSVInputError(f"unsupported CSV encoding: {path.name}")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
