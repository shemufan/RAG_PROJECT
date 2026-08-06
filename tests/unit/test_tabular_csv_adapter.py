import csv

from app.services.tabular_csv_adapter import TabularCSVAdapter


def test_tabular_creates_one_case_per_column_with_bounded_samples(tmp_path):
    path = tmp_path / "ordinary.csv"
    long_value = "x" * 80
    rows = [["user_id", "reg_ip", "empty_field"]]
    rows.extend(
        [str(index), long_value if index == 1 else f"10.0.*.{index}", ""]
        for index in range(1, 8)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    batch = TabularCSVAdapter().load(path)

    assert batch.input_mode == "tabular"
    assert [case.field_profile.field_name for case in batch.cases] == [
        "user_id",
        "reg_ip",
        "empty_field",
    ]
    assert batch.cases[0].field_profile.sample_values == ["1", "2", "3", "4", "5"]
    assert len(batch.cases[1].field_profile.sample_values[0]) == 50
    assert batch.cases[2].field_profile.sample_values == []
    assert all(case.field_profile.database_name == "csv_source" for case in batch.cases)
