from drc_timepoint.runner import run_analysis_from_config
import pandas as pd


def test_full_pipeline_flow(tmp_path):
    # 1. Create a tiny real CSV file
    csv_file = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "Species": ["SF", "SF", "SF", "SF"],  # 4 rows
            "uM": [0, 10, 0, 10],  # 2 doses
            "RawOD": [0.1, 0.5, 0.2, 0.6],  # Growth over time
            "Time": [1.0, 1.0, 2.0, 2.0],  # TWO distinct timepoints
        }
    ).to_csv(csv_file, index=False)

    # 2. Create a config dict
    config = {
        "file_path": str(csv_file),
        "group_fields": ["Species"],
        "dose_field": "uM",
        "od_field": "RawOD",
        "time_field": "Time",
        "top_n": 1,
    }

    # 3. ACTION: Run the whole runner!
    result = run_analysis_from_config(config)

    # 4. ASSERT: Did it reach the end?
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "composite_score" in result.columns
