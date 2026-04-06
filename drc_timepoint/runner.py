import pandas as pd
from .io import read_csv_file
from .validation import match_columns, validate_numeric_columns
from .analysis import (
    standardize_od,
    compute_metrics,
    compute_composite_score,
    select_optimal_timepoint,
)


def run_analysis_from_config(config: dict) -> pd.DataFrame:
    """
    Run the full DRC timepoint selection pipeline from a loaded config dict.

    :param config: Validated configuration dictionary.
    :type config: dict
    :returns: DataFrame containing the optimal timepoint per group.
    :rtype: pandas.DataFrame
    """
    weights = {
        "snr": 0.35,
        "correlation": 0.30,
        "cv": -0.20,
        "dynamic_range": 0.10,
        "smoothness": 0.05,
    }

    FILE_PATH_STR = config["file_path"]
    GROUP_FIELDS = [f.lower() for f in config["group_fields"]]
    DOSE_FIELD = config["dose_field"].lower()
    OD_FIELD = config["od_field"].lower()
    TIME_FIELD = config["time_field"].lower()
    STANDARDIZE_METHOD = config.get("standardize_method", "minmax")

    df = read_csv_file(FILE_PATH_STR)

    required_columns = GROUP_FIELDS + [DOSE_FIELD, OD_FIELD, TIME_FIELD]
    match_columns(df, required_columns)

    validate_numeric_columns(df, [DOSE_FIELD, OD_FIELD])

    df[f"{OD_FIELD}_std"] = standardize_od(df, OD_FIELD, method=STANDARDIZE_METHOD)

    metrics_df = compute_metrics(
        df, GROUP_FIELDS, DOSE_FIELD, TIME_FIELD, f"{OD_FIELD}_std"
    )

    normalized_metrics_df = compute_composite_score(metrics_df, weights)

    best_timepoint_df = select_optimal_timepoint(normalized_metrics_df, GROUP_FIELDS)

    return best_timepoint_df
