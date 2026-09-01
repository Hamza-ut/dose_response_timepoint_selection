import pandas as pd


# Columns matcher
def match_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Ensure required columns exist in the DataFrame.

    :param df: Input DataFrame
    :param required_columns: Columns that must exist
    """
    # standardize column names for case-insensitive matching (without mutating the caller's DataFrame)
    df_columns = [col.strip().lower() for col in df.columns]
    required_columns = [col.strip().lower() for col in required_columns]

    missing = [col for col in required_columns if col not in df_columns]
    if missing:
        raise ValueError(
            "CSV file is missing required column(s) as specified in the config file. \n"
            f"Missing required column(s) from the CSV: {missing}. \n"
        )


# Columns validator
def validate_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """
    Validate numeric columns for non-numeric or missing data.

    :param df: Input DataFrame
    :param numeric_columns: Columns expected to contain numeric data
    """
    # standardize column names for case-insensitive matching (without mutating the caller's DataFrame)
    column_lookup = {col.strip().lower(): col for col in df.columns}
    numeric_columns = [col.strip().lower() for col in numeric_columns]

    for col in numeric_columns:
        # Check 1: Existence
        if col not in column_lookup:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        series = df[column_lookup[col]]

        # Check 2: Numeric Type
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"Column '{col}' must be numeric.")

        # Check 3: Missing Values
        if series.isna().any():
            raise ValueError(f"Column '{col}' contains NaN values. Clean data first.")

        # Check 4: Variance (The 'Identical' check)
        if series.nunique() <= 1:
            raise ValueError(f"Column '{col}' has identical values; cannot process.")
