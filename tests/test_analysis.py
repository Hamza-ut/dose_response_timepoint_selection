# ----------------------------------------------------------------------------- #
# # long way but more readable
# def test_standardize_od_if_od_column_exists():
#     df = pd.DataFrame({"wrong_column": [10.0, 20.0, 30.0]})
#     with pytest.raises(KeyError) as excinfo:
#         standardize_od(df, "raw_od")
#     assert excinfo.type == KeyError
#     assert "not found in dataframe" in str(excinfo.value)

# # short way but less readable
# def test_standardize_od_if_od_column_is_numeric():
#     df = pd.DataFrame({"raw_od": [10.0, 20.0, 30.0, "a"]})
#     with pytest.raises(TypeError, match="Column 'raw_od' must be numeric"):
#         standardize_od(df, "raw_od")


import pytest
import numpy as np
import pandas as pd
from drc_timepoint.analysis import (
    compute_composite_score,
    standardize_od,
    standardize_time,
    compute_metrics,
    get_top_rankings,
)


## using fixture, default scope=function means the fixture is created every time it is used. scope=module means the fixture is created once per test module and reused across all tests
@pytest.fixture(scope="module")
def multi_group_data():
    return pd.DataFrame(
        {
            "Species": ["E.coli"] * 3 + ["S.aureus"] * 3,
            "RawOD": [0.1, 0.5, 0.9, 2.0, 3.0, 4.0],
        }
    )


# Test 1: standardize_od
# use fixture to test that standardize_od correctly scales within groups defined by 'Species'
def test_standardize_od_grouping(multi_group_data):
    result = standardize_od(
        multi_group_data, od_field="RawOD", group_fields=["Species"]
    )

    # Check E.coli group
    ecoli = result[result["Species"] == "E.coli"]
    assert ecoli["RawOD_standardized"].min() == 0.0
    assert ecoli["RawOD_standardized"].max() == 1.0

    # Check S.aureus group
    saureus = result[result["Species"] == "S.aureus"]
    assert saureus["RawOD_standardized"].min() == 0.0
    assert saureus["RawOD_standardized"].max() == 1.0


# Test 2: standardize_od
# test that if a group has only one value, the standardized value is set to 0.0 to avoid division by zero
def test_standardize_od_flatline():
    # If a group has only one value, the range is 0
    df = pd.DataFrame({"Species": ["Flatline"], "RawOD": [0.5]})

    result = standardize_od(df, "RawOD", ["Species"])

    # Your code says it should return 0.0 in this case
    assert result.iloc[0]["RawOD_standardized"] == 0.0


# Test 1: standardize_time
# use parametrize to test that standardize_time correctly rounds time values to the specified precision
@pytest.mark.parametrize(
    "precision, input_time, expected_output",
    [
        (1, 12.05, 12.0),  # Changed to 12.0 because of Banker's Rounding
        (1, 12.06, 12.1),  # 12.06 is clearly closer to 12.1
        (0, 12.6, 13.0),
        (2, 12.123, 12.12),
    ],
)
def test_standardize_time_precisions(precision, input_time, expected_output):
    df = pd.DataFrame({"Time": [input_time]})
    result = standardize_time(df, time_field="Time", precision=precision)

    # Use approx to handle tiny floating point errors
    assert result.iloc[0]["Time_standardized"] == pytest.approx(expected_output)


# Fixture for compute_metrics tests: provides a standardized dataset with known properties to test metric calculations
@pytest.fixture
def baseline_metrics_data():
    """
    Standardized 'Perfect' data for one timepoint:
    - 3 Doses: 0, 10, 100
    - Replicates: 2 per dose (Identical values to keep std_od = 0)
    - Mean ODs: 0.1, 0.5, 0.9 (Perfectly linear increase)
    """
    return pd.DataFrame(
        {
            "species": ["E.coli"] * 6,
            "uM": [0, 10, 100, 0, 10, 100],
            "hour": [5.0] * 6,
            "raw_od": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
        }
    )


# Test 1: compute_composite_score
# test that compute_composite_score correctly normalizes metrics and computes the composite score based on weights
def test_compute_metrics_structure(baseline_metrics_data):
    # ACTION
    results = compute_metrics(
        baseline_metrics_data,
        group_fields=["species"],
        dose_field="uM",
        time_field="hour",
        od_field="raw_od",
    )

    # ASSERT structure
    assert len(results) == 1  # We only provided one timepoint (5.0)
    assert "species" in results.columns
    assert "hour" in results.columns
    # Ensure all 5 metrics are present as keys[cite: 2]
    for metric in ["snr", "correlation", "cv", "dynamic_range", "smoothness"]:
        assert metric in results.columns


# Test 2: compute_composite_score
# test that compute_composite_score correctly normalizes metrics and computes the composite score based on weights
def test_compute_metrics_math_logic(baseline_metrics_data):
    results = compute_metrics(
        baseline_metrics_data,
        group_fields=["species"],
        dose_field="uM",
        time_field="hour",
        od_field="raw_od",
    )
    row = results.iloc[0]

    # 1. Correlation: Should be 1.0 because 0.1 < 0.5 < 0.9 matches 0 < 10 < 100[cite: 2]
    assert row["correlation"] == pytest.approx(1.0)

    # 2. SNR: average_plate_noise is 0 (identical replicates), so SNR defaults to 0[cite: 2]
    assert row["snr"] == 0

    # 3. CV: (std / mean). Since std is 0, CV must be 0[cite: 2]
    assert row["cv"] == pytest.approx(0.0)

    # 4. Dynamic Range: log10((0.9 + 1e-4) / (0.1 + 1e-4)) ≈ log10(9) ≈ 0.954[cite: 2]
    assert row["dynamic_range"] == pytest.approx(0.954, abs=1e-3)

    # 5. Smoothness: Optimal jump for 2 intervals is 0.25. Our jump is 0.4.[cite: 2]
    # It won't be 1.0, but it should be a valid float between 0 and 1.
    assert 0 < row["smoothness"] <= 1.0


# Test 1: compute_composite_score
def test_compute_composite_score_weight_inversion():
    # SETUP: Two timepoints, one has higher CV (bad)
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "cv": [0.1, 0.8],  # 0.1 is "better"
            "snr": [10.0, 10.0],  # SNR is identical
        }
    )

    # ACTION: Give CV a negative weight
    weights = {"cv": -1.0, "snr": 1.0}
    result = compute_composite_score(df, weights)

    # ASSERT: Time 1.0 (lower CV) MUST have a higher score than Time 2.0
    assert (
        result.loc[result["time"] == 1.0, "composite_score"].iloc[0]
        > result.loc[result["time"] == 2.0, "composite_score"].iloc[0]
    )


# Test1: get_top_rankings
def test_get_top_rankings_window_logic():
    # SETUP: One timepoint at 5.0 with precision 1 (0.1 steps)
    df = pd.DataFrame(
        {
            "species": ["E.coli"],
            "hour": [5.0],
            "composite_score": [0.95],
            "snr": [10],
            "correlation": [1],
            "cv": [0.1],
            "dynamic_range": [2],
            "smoothness": [1],
        }
    )

    # ACTION: Get ranking with precision 1
    result = get_top_rankings(
        df, group_fields=["species"], time_field="hour", precision=1
    )

    # ASSERT: Check the window string calculation
    # Precision 1 means step is 0.1, buffer is 0.05.
    # Window = (5.0 - 0.05) to (5.0 + 0.05) -> "4.95 to 5.05"
    assert result.iloc[0]["ideal_time_window"] == "4.95 to 5.05"
