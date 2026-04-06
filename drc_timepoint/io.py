import json
import pandas as pd
from pathlib import Path


# JSON config file loader
def load_config(config_path: str | Path) -> dict:
    """
    Load and validate a JSON configuration file which should contain necessary keys.
    """
    config_path = Path(config_path)
    # check1: valid file path must exist
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    # check2: only json files are allowed
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"Config file must be a JSON file: {config_path}")
    # check3: json formatting should not be off
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in config file: {e}")
    # check4: json file must have the following keys, not any random json
    required_keys = [
        "file_path",
        "group_fields",
        "dose_field",
        "od_field",
        "time_field",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing key(s) in config: {missing}")

    return config


# CSV Reader
def read_csv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file and create pandas df.

    :param file_path: Path to the CSV file.
    :type file_path: str or Path
    :returns: Loaded DataFrame.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If file is missing.
    :raises ValueError: If file is empty or malformed.
    """
    file_path = Path(file_path)
    # Check 1: file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"File '{file_path}' does not exist. Check your config file for the correct path."
        )
    # Check 2: only CSV allowed
    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Only .csv files are allowed: '{file_path.name}' not a CSV file."
        )
    # Try reading CSV with robust error handling
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file '{file_path}' is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Failed to parse CSV file '{file_path}': {e}. "
            "Possible causes: bad delimiter, inconsistent columns, or malformed data."
        )
    except Exception as e:
        raise ValueError(f"CSV Reader Error for file '{file_path}': {e}")
    # Check 3: empty DataFrame after reading
    if df.empty:
        raise ValueError(f"CSV file '{file_path}' contains no data.")

    return df
