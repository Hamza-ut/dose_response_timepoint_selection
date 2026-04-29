import pandas as pd
from .io import read_csv_file
from .validation import match_columns, validate_numeric_columns
from .analysis import (
    standardize_od,
    standardize_time,
    compute_metrics,
    compute_composite_score,
    get_top_rankings,
)


def run_analysis_from_config(config: dict) -> pd.DataFrame:
    """
    Run the full DRC timepoint selection pipeline from a loaded config dict.

    :param config: Validated configuration dictionary
    :returns: DataFrame containing the optimal timepoint per group
    """
    weights = {
        "snr": 0.30,  # Positive: High SNR is Good
        "correlation": 0.30,  # Positive: High Correlation is Good
        "cv": -0.25,  # Negative: High CV is BAD (High noise)
        "dynamic_range": 0.05,  # Positive: High Range is Good
        "smoothness": 0.10,  # Positive: High Smoothness is Good
    }

    FILE_PATH_STR = config["file_path"]
    GROUP_FIELDS = [f.lower() for f in config["group_fields"]]
    DOSE_FIELD = config["dose_field"].lower()
    OD_FIELD = config["od_field"].lower()
    TIME_FIELD = config["time_field"].lower()
    TOP_N = config.get("top_n", 3)

    required_columns = GROUP_FIELDS + [DOSE_FIELD, OD_FIELD, TIME_FIELD]

    df = read_csv_file(FILE_PATH_STR)

    # run1: Validate that required columns are present
    match_columns(df, required_columns)

    # run2: Validate that dose, OD, and time columns contain numeric data
    validate_numeric_columns(df, [DOSE_FIELD, OD_FIELD, TIME_FIELD])

    # run3: Standardize od values per group
    df = standardize_od(df, OD_FIELD, group_fields=GROUP_FIELDS)

    # run4: Standardize time values to avoid machine deflections due to small time differences
    df = standardize_time(df, TIME_FIELD, precision=1)

    # run5: Compute metrics for each group and timepoint
    metrics_df = compute_metrics(
        df,
        GROUP_FIELDS,
        DOSE_FIELD,
        f"{TIME_FIELD}_standardized",
        f"{OD_FIELD}_standardized",
    )

    # run6: Compute composite score using the specified weights
    composite_score_df = compute_composite_score(metrics_df, weights)

    # run7: Get top ranking timepoint per group based on composite score
    top_n_timepoints = get_top_rankings(
        composite_score_df,
        GROUP_FIELDS,
        f"{TIME_FIELD}_standardized",  # Fix: Use the column that actually exists in metrics_df
        top_n=TOP_N,
    )

    return top_n_timepoints
