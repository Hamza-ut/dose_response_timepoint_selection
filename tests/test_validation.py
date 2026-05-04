from drc_timepoint.validation import match_columns, validate_numeric_columns
import pandas as pd
import pytest


# ------- TESTS FOR match_columns ------- #
# Test 1: Case-insensitive column matching
def test_match_columns_lower_case_and_match(csv_data, required_columns):
    match_columns(csv_data, required_columns)


# Test 2: Missing required column
def test_match_columns_missing_column(csv_data, required_columns):
    with pytest.raises(ValueError, match="Missing required column"):
        match_columns(csv_data.drop(columns=["HOUR"]), required_columns)


# ------- TESTS FOR validate_numeric_columns ------- #
# Test 1: numeric columns exists
def test_validate_numeric_columns_valid(csv_data):
    required_columns = ["HOUR", "RaW_Od"]
    validate_numeric_columns(csv_data, required_columns)


# Test 2: error cases for validate_numeric_columns
@pytest.mark.parametrize(
    "data, cols_to_check, error_type, match_msg",
    [
        # Case 1: Column doesn't exist
        ({"real_col": [1, 2]}, ["ghost_col"], KeyError, "not found"),
        # Case 2: Column is strings, not numbers
        ({"species": ["A", "B"]}, ["species"], TypeError, "must be numeric"),
        # Case 3: Column has a missing value (NaN)
        ({"dose": [1.0, None]}, ["dose"], ValueError, "contains NaN values"),
        # Case 4: Column has identical values (no variance)[cite: 1]
        ({"xmic": [128, 128, 128]}, ["xmic"], ValueError, "has identical values"),
    ],
)
def test_validate_numeric_errors(data, cols_to_check, error_type, match_msg):
    df = pd.DataFrame(data)
    with pytest.raises(error_type, match=match_msg):
        validate_numeric_columns(df, cols_to_check)
