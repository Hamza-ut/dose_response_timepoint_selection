import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Standardize OD values, can be used as standalone function elsewhere
def standardize_od(
    df: pd.DataFrame, od_field: str, method: Literal["zscore", "minmax"] = "minmax"
) -> pd.Series:
    """
    Standardize OD values .

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param od_field: Column containing OD values.
    :type od_field: str
    :param method: Standardization method.
    :type method: Literal["zscore", "minmax"]
    :returns: Standardized OD series.
    :rtype: pandas.Series
    :raises ValueError: For invalid method or bad data.
    """
    # check 1: od_field exists
    if od_field not in df.columns:
        raise KeyError(f"Column '{od_field}' not found in dataframe.")
    series = df[od_field]
    # check 2: od_field must be numeric, non-NaN and have variance
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{od_field}' must be numeric for standardization.")
    if series.isna().any():
        raise ValueError(
            f"Column '{od_field}' contains NaN values. Please clean data first."
        )
    if series.nunique() <= 1:
        raise ValueError(f"Cannot standardize '{od_field}': all values are identical.")
    # Choose scaler
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            "Invalid standardization method: choose 'zscore' or 'minmax'. Check your config file for typo."
        )
    # sklearn expects a 2D array
    scaled = scaler.fit_transform(series.to_frame())
    return pd.Series(
        scaled.flatten(), index=series.index, name=f"{od_field}_standardized"
    )


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

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param group_fields: Columns to group by.
    :type group_fields: List[str]
    :param dose_field: Dose column.
    :type dose_field: str
    :param time_field: Time column.
    :type time_field: str
    :param od_field: OD column.
    :type od_field: str
    :returns: DataFrame of computed metrics.
    :rtype: pandas.DataFrame
    """
    # Initial grouping to aggregate rawOD for all replicates (group by group_fields + dose + time → computes mean and std for each replicate set)
    grouped = (
        df.groupby(group_fields + [dose_field, time_field])[od_field]
        .agg(mean_od="mean", std_od="std", count="count")
        .reset_index()
    )
    # Grouping by timepoint (loop over timepoints within each group)
    results = []
    for name, group in grouped.groupby(group_fields + [time_field]):
        group_info = dict(zip(group_fields + [time_field], name))
        # 1. SNR (signal / noise)
        max_signal = group["mean_od"].max() - group["mean_od"].min()
        avg_noise = group["std_od"].mean()
        snr = max_signal / avg_noise if avg_noise != 0 else 0
        # 2. Spearman correlation of dose vs mean OD
        correlation_abs = abs(
            stats.spearmanr(group[dose_field], group["mean_od"]).correlation
        )
        # 3. Coefficient of variation (The small number 1e-8 is added to avoid dividing by zero in case any mean_od happens to be 0)
        cv = (group["std_od"] / (group["mean_od"] + 1e-8)).mean()
        # 4. Dynamic range (if min mean_od is 0, the result is set to 0 to avoid division-by-zero errors)
        dynamic_range = (
            group["mean_od"].max() / group["mean_od"].min()
            if group["mean_od"].min() != 0
            else 0
        )
        # 5. Smoothness (mean absolute difference between successive doses). It is standard in dose–response analysis, often called “slope consistency” or “smoothness metric.
        n_doses = len(group["mean_od"])
        # number of interval
        n_intervals = n_doses - 1
        # 0.5 is half the normalized range of OD (assuming OD is scaled 0–1). Dividing by intervals gives ideal size of jump per step
        optimal = 0.5 / (n_intervals)
        # Tolerance value controls how quickly the score drops off as you move away from the optimal MAD.
        tolerance = 0.1
        # Measure how much the curve jumps between consecutive doses, on average.
        mad = np.abs(np.diff(group["mean_od"])).mean()
        # Gaussian-like mapping of MAD to 0–1. When MAD == optimal, smoothness = 1. As MAD deviates from optimal, smoothness decreases.
        smoothness = np.exp(-((mad - optimal) ** 2) / (2 * (tolerance**2)))

        # zip all metrics together
        results.append(
            {
                **group_info,
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

    :param df: DataFrame containing metric columns.
    :type df: pandas.DataFrame
    :param weights: Metric weights; negative values invert normalization.
    :type weights: Dict[str, float]
    :returns: DataFrame with normalized metrics and composite score.
    :rtype: pandas.DataFrame
    :raises KeyError: If metric columns are missing.
    """
    # apply min max normalization to the resultant metrics to make it all homogenous.
    for metric, weight in weights.items():
        col_norm = f"{metric}_norm"
        df[col_norm] = (df[metric] - df[metric].min()) / (
            df[metric].max() - df[metric].min()
        )
        # since lower CV and smoothness is better so that's why we gave negative weight to use it to flip the results
        if weight < 0:
            # This flip makes every normalized metric’s direction the same
            df[col_norm] = 1 - df[col_norm]
    df["composite_score"] = sum(
        df[f"{metric}_norm"] * abs(weight) for metric, weight in weights.items()
    )
    return df


# Select optimal timepoint per group
def select_optimal_timepoint(df: pd.DataFrame, group_fields: List[str]) -> pd.DataFrame:
    """
    Select rows with maximum composite score per group.

    :param df: Input DataFrame.
    :type df: pandas.DataFrame
    :param group_fields: Grouping columns.
    :type group_fields: List[str]
    :returns: Optimal rows for each group.
    :rtype: pandas.DataFrame
    :raises ValueError: If grouping columns are missing.
    """
    # returns the index of the row with the maximum value in each group.
    idx = df.groupby(group_fields)["composite_score"].idxmax()
    # selects rows by index labels and return that
    return df.loc[idx].reset_index(drop=True)
