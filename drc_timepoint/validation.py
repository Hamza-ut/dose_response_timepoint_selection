import pandas as pd


# Columns matcher
def match_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Ensure required columns exist in the DataFrame.

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param required_columns: Columns that must exist.
    :type required_columns: list[str]
    :raises ValueError: If required columns are missing.
    """
    df_columns = [col.strip().lower() for col in df.columns]
    missing = [col for col in required_columns if col.lower() not in df_columns]
    if missing:
        raise ValueError(
            "CSV file is missing required column(s) as specified in the config file. \n"
            f"Missing required column(s) from the CSV: {missing}. \n"
        )


# Columns validator
def validate_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> None:
    """
    Validate numeric columns for non-numeric or missing data.

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param numeric_columns: Columns expected to contain numeric data.
    :type numeric_columns: list[str]
    :raises KeyError: If column is missing.
    :raises TypeError: If non-numeric values are detected.
    :raises ValueError: If NaN values are present.
    """
    for col in numeric_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        series = df[col]
        # Check for NaNs
        if series.isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values. "
                "Please clean or fill missing data."
            )
        # Check for non-numeric values
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            non_numeric_mask = coerced.isna()
            if non_numeric_mask.any():
                bad_values = series[non_numeric_mask].unique()
                n_bad = len(bad_values)
                preview = ", ".join(map(str, bad_values[:5]))
                if n_bad > 5:
                    preview += f", ... (+{n_bad - 5} more)"
                raise TypeError(
                    f"Column '{col}' contains {n_bad} non-numeric values "
                    f"(e.g., {preview}). Ensure all entries are numeric."
                )
