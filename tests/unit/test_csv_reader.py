import csv

import pytest

from app.schemas.csv_input import CSVReadLimits
from app.services.csv_reader import CSVInputError, CSVReader


def write_csv(path, rows, *, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as handle:
        csv.writer(handle).writerows(rows)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-8", "gb18030"])
def test_reader_validates_encoding_and_streams_rows(tmp_path, encoding):
    path = tmp_path / "字段.csv"
    write_csv(path, [[" 字段名 ", "样本1"], ["手机号", "138****"]], encoding=encoding)
    reader = CSVReader()

    inspection = reader.inspect(path)
    rows = list(reader.iter_rows(path, inspection))

    assert inspection.headers == ["字段名", "样本1"]
    assert inspection.source_name == "字段.csv"
    assert inspection.row_count == 1
    assert len(inspection.source_fingerprint) == 64
    assert rows == [(2, {"字段名": "手机号", "样本1": "138****"})]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "empty"),
        ([['', 'value']], "blank header"),
        ([['name', ' name ']], "duplicate header"),
    ],
)
def test_reader_rejects_invalid_headers(tmp_path, rows, message):
    path = tmp_path / "invalid.csv"
    write_csv(path, rows)

    with pytest.raises(CSVInputError, match=message):
        CSVReader().inspect(path)


def test_reader_enforces_size_row_and_column_limits(tmp_path):
    path = tmp_path / "limited.csv"
    write_csv(path, [["a", "b"], ["1", "2"], ["3", "4"]])

    with pytest.raises(CSVInputError, match="size"):
        CSVReader(CSVReadLimits(max_bytes=1)).inspect(path)
    with pytest.raises(CSVInputError, match="columns"):
        CSVReader(CSVReadLimits(max_columns=1)).inspect(path)
    with pytest.raises(CSVInputError, match="rows"):
        CSVReader(CSVReadLimits(max_rows=1)).inspect(path)


def test_reader_rejects_rows_with_wrong_column_count(tmp_path):
    path = tmp_path / "ragged.csv"
    write_csv(path, [["a", "b"], ["1"]])

    with pytest.raises(CSVInputError, match="row 2"):
        CSVReader().inspect(path)
