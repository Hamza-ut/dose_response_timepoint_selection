import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict


# Standardize OD values
def standardize_od(
    df: pd.DataFrame, od_field: str, group_fields: List[str] = None
) -> pd.DataFrame:
    """
    Standardize OD values to min-max scale for each group defined by group_fields.

    :param df: Input DataFrame
    :param od_field: Column containing OD values
    :param group_fields: (Optional) Columns to group by before scaling
    :returns: DataFrame with an added '{od_field}_standardized' column
    """
    df = df.copy()

    # Calculate min and max for each specific group
    mins = df.groupby(group_fields)[od_field].transform("min")
    maxs = df.groupby(group_fields)[od_field].transform("max")
    ranges = maxs - mins

    # Apply scaling with safety check for flat-line data
    df[f"{od_field}_standardized"] = np.where(
        ranges == 0, 0.0, (df[od_field] - mins) / ranges
    )

    return df


# Standardize time by rounding to a specific decimal precision. This handles micro-drift (seconds) without collapsing real measurements.
def standardize_time(
    df: pd.DataFrame, time_field: str, precision: int = 1
) -> pd.DataFrame:
    """
    Standardizes time by rounding to a specific decimal precision.

    :param df: Input DataFrame
    :param time_field: Column containing time values
    :param precision: 1 = nearest 6 mins, 2 = nearest 36 seconds
    :returns: DataFrame with standardized time column
    """
    df = df.copy()
    # 6-minute gravity buckets to round machine drift
    df[f"{time_field}_standardized"] = df[time_field].round(precision)
    return df


# Core function to compute metrics
def compute_metrics(
    df: pd.DataFrame,
    group_fields: List[str],
    dose_field: str,
    time_field: str,
    od_field: str,
) -> pd.DataFrame:
    """
    Compute SNR, correlation, CV, dynamic range, and smoothness.

    :param df: Input DataFrame
    :param group_fields: Columns to group by (e.g., species, condition)
    :param dose_field: Dose column
    :param time_field: Time column
    :param od_field: OD column
    :returns: DataFrame of computed metrics
    """
    # STEP 1: Summarize Replicates (The "Dose-Level" Summary)
    # We collapse the plates into a single average and standard deviation for every specific dose for every group.
    dose_summaries = (
        df.groupby(group_fields + [dose_field, time_field])[od_field]
        .agg(mean_od="mean", std_od="std", count="count")
        .reset_index()
    )

    results = []

    # STEP 2: Analyze the Curve (The "Timepoint-Level" Summary)
    # We grab all the doses for a single timepoint to see how the whole curve looks at that specific timepoint
    for group_values, curve_table in dose_summaries.groupby(
        group_fields + [time_field]
    ):

        # STEP 3:Extract metadata (Species, Time, etc.), this will be used to label the metrics for this specific timepoint
        timepoint_metadata = dict(zip(group_fields + [time_field], group_values))

        ## METRIC 1: SNR (Signal-to-Noise Ratio)
        # 1.1: Measure the Curve (The 'Signal')
        max_response = curve_table["mean_od"].max()
        min_response = curve_table["mean_od"].min()
        curve_delta = max_response - min_response
        # 1.2: Measure the Jitter/noise (average the plate-to-plate disagreement across all doses)
        average_plate_noise = curve_table["std_od"].mean()
        # 1.3: Compute SNR (signal / noise), with a check to avoid division by zero
        snr = curve_delta / average_plate_noise if average_plate_noise != 0 else 0

        # METRIC 2: Spearman correlation of dose vs mean OD
        if curve_table["mean_od"].nunique() <= 1:
            correlation_abs = 0
        else:
            # we are discarding the p-value because we are only interested in strength
            rho, _ = stats.spearmanr(curve_table[dose_field], curve_table["mean_od"])
            correlation_abs = abs(rho)

        # METRIC 3: Coefficient of variation for every dose (The small number 1e-8 is added to avoid dividing by zero in case any mean_od happens to be 0)
        cv = (curve_table["std_od"] / (curve_table["mean_od"] + 1e-8)).mean()

        # METRIC 4: Dynamic range (if min mean_od is 0, the result is set to 0 to avoid division-by-zero errors)
        dynamic_range = curve_table["mean_od"].max() / (
            curve_table["mean_od"].min() + 1e-8
        )

        # METRIC 5: Smoothness or slope consistency (mean absolute difference between successive doses).
        n_doses = len(curve_table["mean_od"])
        # number of intervals will always be one less than number of doses
        n_intervals = n_doses - 1
        # 0.5 is half the normalized range of OD (assuming OD is scaled 0–1). Dividing by intervals gives ideal size of jump per step
        # A perfect experiment is one where the drug inhibits 50% of growth at the middle dose, so the curve should ideally jump from 0 to 0.5 in the first half of the doses,
        # and then from 0.5 to 1 in the second half. This means that for a perfectly smooth curve, the average jump between doses should be around 0.5 divided by the number of intervals.
        optimal = 0.5 / (n_intervals)
        # Tolerance value controls how quickly the score drops off as you move away from the optimal MAD.
        tolerance = 0.1
        # Measure how much the curve jumps between consecutive doses, on average.
        mad = np.abs(np.diff(curve_table["mean_od"])).mean()
        # Gaussian-like mapping of MAD to 0–1. When MAD == optimal, smoothness = 1. As MAD deviates from optimal, smoothness decreases.
        smoothness = np.exp(-((mad - optimal) ** 2) / (2 * (tolerance**2)))

        # zip all metrics together
        results.append(
            {
                **timepoint_metadata,
                "snr": snr,
                "correlation": correlation_abs,
                "cv": cv,
                "dynamic_range": dynamic_range,
                "smoothness": smoothness,
            }
        )
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# Normalize metrics to compute composite score
def compute_composite_score(
    df: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Normalize metrics and compute composite scores.

    :param df: DataFrame containing metric columns
    :param weights: Metric weights; negative values invert normalization
    :returns: DataFrame with normalized metrics and composite score
    """
    # min max scaling for every metric.
    for metric, weight in weights.items():
        col_norm = f"{metric}_norm"

        # handle edge cases where all values are the same (range = 0) to avoid division by zero
        range_val = df[metric].max() - df[metric].min()
        if range_val == 0:
            df[col_norm] = (
                1.0 if weight > 0 else 0.0
            )  # Safe default if all values are identical
        else:
            df[col_norm] = (df[metric] - df[metric].min()) / range_val

        # since lower CV is better so that's why we gave negative weight in dictionary to use it to flip the results
        if weight < 0:
            # This flip makes every normalized metric’s direction the same
            df[col_norm] = 1 - df[col_norm]
    df["composite_score"] = sum(
        df[f"{metric}_norm"] * abs(weight) for metric, weight in weights.items()
    )
    return df


def get_top_rankings(
    metrics_df: pd.DataFrame,
    group_fields: list,
    time_field: str,
    top_n: int = 3,
    precision: int = 1,
):

    # 1. Sort by score
    df_sorted = metrics_df.sort_values(
        by=group_fields + ["composite_score"],
        ascending=[True] * len(group_fields) + [False],
    )

    # 2. Grab top 3
    top_df = df_sorted.groupby(group_fields).head(top_n).copy()

    # 3. Add Rank
    top_df["rank"] = top_df.groupby(group_fields).cumcount() + 1

    # 4. Calculate the Window based on the precision used in standardization
    # If precision is 1, the step is 0.1, so the "reach" is 0.05 on either side.
    interval_step = 10**-precision
    buffer = interval_step / 2

    # Create the window string: e.g., "12.95 - 13.05"
    top_df["ideal_time_window"] = (
        (top_df[time_field] - buffer).round(2).astype(str)
        + " to "
        + (top_df[time_field] + buffer).round(2).astype(str)
    )

    raw_metric_names = ["snr", "correlation", "cv", "dynamic_range", "smoothness"]
    top_df = top_df[
        [
            *group_fields,
            "ideal_time_window",
            "rank",
            "composite_score",
            *raw_metric_names,
        ]
    ]

    return top_df


# ----------------------------------------------------------- #
# FILE_PATH_STR = r"data\timepoint_vallo.csv"
# df = pd.read_csv(FILE_PATH_STR)
# GROUP_FIELDS = ["Species"]
# DOSE_FIELD = "uM"
# OD_FIELD = "RawOD"
# TIME_FIELD = "Time_h"
# TOP_N = 3

# FILE_PATH_STR = r"data\timepoint_sf.csv"
# df = pd.read_csv(FILE_PATH_STR)
# GROUP_FIELDS = ["Condition", "Ratio"]
# DOSE_FIELD = "XMIC"
# OD_FIELD = "Raw_od"
# TIME_FIELD = "hour"
# TOP_N = 2
