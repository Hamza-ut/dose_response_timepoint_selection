from drc_timepoint.io import load_config, read_csv_file
from pathlib import Path
import pandas as pd
import pytest
import json


# ------- TESTS FOR load_config ------- #
# Test 1: load_config success case, json must have all mandatory keys
def test_load_config_with_all_mandatory_keys(tmp_path):
    # 1. SETUP
    valid_data = {
        "file_path": "data/timepoint_vallo.csv",
        "group_fields": ["Species"],
        "dose_field": "uM",
        "od_field": "RawOD",
        "time_field": "Time_h",
        "top_n": 2,
        "export_results": False,
    }
    config_file = tmp_path / "valid_config.json"
    config_file.write_text(json.dumps(valid_data))

    result = load_config(config_file)

    assert isinstance(result, dict)
    required = ["file_path", "group_fields", "dose_field", "od_field", "time_field"]
    for key in required:
        assert key in result

    # Verify types were preserved during the JSON -> Python roundtrip
    assert isinstance(result["export_results"], bool)
    assert result["top_n"] == 2


# Test 2: load config file not found
def test_load_config_file_not_found(tmp_path):
    non_existent_file = tmp_path / "nonexistent_config.json"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(non_existent_file)


# Test 3: load_config with remaining error cases (must be json, must have required keys, must be valid json)
@pytest.mark.parametrize(
    "filename, content, error_type, match_msg",
    [
        # Check 2: Wrong suffix
        ("config.txt", {"file_path": "data.csv"}, ValueError, "must be a JSON file"),
        # Check 3: Bad JSON format (we pass a string that isn't valid JSON)
        ("config.json", "invalid json {", ValueError, "Invalid JSON format"),
        # Check 4: Missing required keys
        ("config.json", {"random_key": 123}, KeyError, "Missing key"),
    ],
)
def test_load_config_errors(tmp_path, filename, content, error_type, match_msg):
    # 1. SETUP: Create a temporary file
    test_file = tmp_path / filename

    # If content is a dict, write it as JSON; if string, write raw text
    if isinstance(content, dict):
        test_file.write_text(json.dumps(content))
    else:
        test_file.write_text(content)

    # 2. ACTION & ASSERT
    with pytest.raises(error_type, match=match_msg):
        load_config(test_file)


# ------- TESTS FOR read_csv_file ------- #
# Test 1: read_csv_file success case with the .str.lower() logic
def test_read_csv_file_lower_case(tmp_path, csv_data, required_columns):
    # 1. SETUP: Create a real file from the fixture data
    file_path = tmp_path / "test_data.csv"
    csv_data.to_csv(file_path, index=False)  # Physically save it!

    # 2. ACTION: Now read_csv_file gets the path it wants
    df = read_csv_file(file_path)

    # 3. ASSERT: Check if your .str.lower() logic worked
    expected_cols = required_columns
    assert df.columns.tolist() == expected_cols


# Test 2: read_csv_file error case with empty file (after reading, we make it empty to trigger the error)
@pytest.mark.parametrize(
    "filename, content, error_type, match_msg, create_file",
    [
        # 1. Path doesn't exist
        ("ghost.csv", "", FileNotFoundError, "does not exist", False),
        # 2. Wrong Extension
        ("wrong.txt", "", ValueError, "Only .csv files are allowed", True),
        # 3. Empty File (0 bytes)
        ("empty.csv", "", ValueError, "is empty", True),
        # 4. Headers but no data (df.empty)
        ("headers_only.csv", "col1,col2", ValueError, "contains no data", True),
    ],
)
def test_read_csv_all_errors(
    tmp_path, filename, content, error_type, match_msg, create_file
):
    test_file = tmp_path / filename

    if create_file:
        test_file.write_text(content)

    with pytest.raises(error_type, match=match_msg):
        read_csv_file(test_file)
