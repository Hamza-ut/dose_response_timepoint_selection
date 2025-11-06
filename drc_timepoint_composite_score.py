## DEPENDENCIES
import sys
import json
import timeit
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


## FUNCTIONS
# JSON config file loader
def load_config(config_path):
    """Loads and validates the JSON config file."""
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
        raise KeyError(f"Missing keys in config: {missing}")

    return config


# CSV Reader
def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    file_path = Path(file_path)
    # check1: path exists
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    # check2: only csv files are allowed
    if file_path.suffix.lower() != ".csv":
        raise ValueError("Input file must be a .csv file.")
    # check3: robust pandas parsing and empty file check
    try:
        # Try normal read first
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError as e:
        # Often caused by bad delimiters or malformed lines
        raise ValueError(
            f"Failed to parse CSV file '{file_path}': {e}. "
            "Possible causes: bad delimiter, inconsistent columns, or malformed data."
        )
        # catching error for empty csv file
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file '{file_path}' is empty.")

    except Exception as e:
        raise ValueError(f"Unexpected error while reading CSV '{file_path}': {e}")


# Columns validator
def validate_columns(df, required_columns):
    """Validates that all required columns are present in the DataFrame."""
    df_columns = [col.strip().lower() for col in df.columns]
    missing = [col for col in required_columns if col.lower() not in df_columns]
    if missing:
        raise ValueError(
            "Please ensure that your CSV file includes all columns listed in your json config file. \n"
            f"Missing required columns from the CSV: {missing}. \n"
            f"CSV columns found: {list(df.columns)}."
        )


# Numeric columns validator
def validate_numeric_columns(df, numeric_columns):
    """Validates that specified columns contain only numeric values.

    Raises informative errors if non-numeric or NaN values are found.
    """
    for col in numeric_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        series = df[col]

        # Check for NaNs
        if series.isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values. "
                "Please clean or fill missing data before analysis."
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


# Standardize OD values
def standardize_od(df, od_field, method="minmax"):
    """Standardize OD values using z-score or min-max normalization.

    Fails gracefully if:
        - The column doesn't exist.
        - The column contains NaNs or non-numeric values.
        - The column has no variance.
    """
    # check1: column must exist
    if od_field not in df.columns:
        raise KeyError(f"Column '{od_field}' not found in dataframe.")

    series = df[od_field]
    # check2: missing values
    if series.isna().any():
        raise ValueError(
            f"Column '{od_field}' contains NaN values. "
            "Please clean or fill missing data before standardization."
        )
    # check3: ensure variance
    if series.nunique() <= 1:
        raise ValueError(
            f"Cannot standardize '{od_field}': all values are identical (no variance)."
        )

    # perform standardization
    if method == "zscore":
        return (series - series.mean()) / series.std(ddof=0)
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    else:
        raise ValueError(
            f"Invalid standardization method '{method}'. Use 'zscore' or 'minmax'."
        )


# Core function to compute metrics
def compute_metrics(df, group_fields, dose_field, time_field, od_field):
    """
    Compute SNR, correlation, CV, dynamic range, smoothness for each group/time.
    Parameters:
        df (pd.DataFrame): Input dataframe (OD column already validated/standardized)
        group_fields (list of str): Columns to group by (e.g., ['Species'])
        dose_field (str): Column name for dose
        time_field (str): Column name for time
        od_field (str): Column name for OD values

    Returns:
        pd.DataFrame: Each row is a group/timepoint with all metrics computed
    """
    # Initial grouping to aggregate rawOD for all replicates
    grouped = (
        df.groupby(group_fields + [dose_field, time_field])[od_field]
        .agg(mean_od="mean", std_od="std", count="count")
        .reset_index()
    )

    # Grouping by timepoint
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
        # 5. Smoothness (mean absolute difference between successive doses)
        smoothness = np.abs(np.diff(group["mean_od"])).mean()

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

    metrics_df = pd.DataFrame(results)
    return metrics_df


# Normalize metrics to compute composite score
def normalize_and_weight(df, weights):
    """Normalizes metrics and applies weights to compute composite score."""
    # apply min max normalization to the resultant metrics to make it all homogenous.
    for metric, weight in weights.items():
        col_norm = f"{metric}_norm"
        df[col_norm] = (df[metric] - df[metric].min()) / (
            df[metric].max() - df[metric].min()
        )
        # since lower CV is better so that's why we gave negative weight to use it to flip the results
        if weight < 0:
            # This flip makes every normalized metric’s direction the same
            df[col_norm] = 1 - df[col_norm]
    df["composite_score"] = sum(
        df[f"{metric}_norm"] * abs(weight) for metric, weight in weights.items()
    )
    return df


# Select best timepoint per group
def get_best_timepoints(df, group_fields, time_field):
    """Select best timepoint per group."""
    # returns the index of the row with the maximum value in each group.
    idx = df.groupby(group_fields)["composite_score"].idxmax()
    # selects rows by index labels and return that
    return df.loc[idx].reset_index(drop=True)


def timepoint_composite_score(
    df,
    group_fields,
    dose_field,
    od_field,
    time_field,
    standardize=False,
    method="minmax",
):
    """Determine best timepoint using composite scoring of multiple metrics."""
    # Standardize OD values using standardize_od function
    if standardize:
        df[od_field] = standardize_od(df, od_field, method)

    # Compute metrics for each group/timepoint
    metrics_df = compute_metrics(df, group_fields, dose_field, time_field, od_field)

    # hamza old weights dictionary
    # weights = {
    #     "snr": 0.3,
    #     "correlation": 0.3,
    #     "cv": -0.2,  # Negative because lower is better and we can invert the normalization results
    #     "dynamic_range": 0.1,
    #     "smoothness": 0.1,
    # }

    # Weights dictionary, Order of priority according to research articles [SNR, Correlation, CV, Dynamic range, smoothness]
    weights = {
        "snr": 0.35,
        "correlation": 0.30,
        "cv": -0.20,  # Negative because lower is better and we can invert the normalization results
        "dynamic_range": 0.10,
        "smoothness": -0.05,
    }

    # Normalize metrics and compute composite score
    normalized_metrics_df = normalize_and_weight(metrics_df, weights)

    # Select best timepoint per group
    best_timepoint_df = get_best_timepoints(
        normalized_metrics_df, group_fields, time_field
    )

    # optional export to csv
    # best_timepoint_df.to_csv("best_timepoints_composite_score.csv", index=False)
    return best_timepoint_df


def main():
    """
    Main entry point for the dose-response analysis program.
    Accepts a JSON config file as a command-line argument and computes
    the best timepoint using composite scoring of multiple metrics.
    """

    # ----------------------------
    # 1. Parse and validate CLI args
    # ----------------------------
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <config_file.json>")
        sys.exit(1)

    config_file = Path(sys.argv[1])
    print(f"\n🔧 Using configuration file: {config_file}")

    try:
        # ----------------------------
        # 2. Load and validate config
        # ----------------------------
        config = load_config(config_file)
        print("✅ Configuration loaded successfully.\n")

        # Extract values from config
        FILE_PATH_STR = config["file_path"]
        GROUP_FIELDS = config["group_fields"]
        DOSE_FIELD = config["dose_field"]
        OD_FIELD = config["od_field"]
        TIME_FIELD = config["time_field"]

        # ----------------------------
        # 3. Read CSV file
        # ----------------------------
        print(f"📄 Reading data file: {FILE_PATH_STR}. \n")
        df = read_csv_file(FILE_PATH_STR)

        # ----------------------------
        # 4. Validate CSV columns
        # ----------------------------
        required_columns = GROUP_FIELDS + [DOSE_FIELD, OD_FIELD, TIME_FIELD]
        validate_columns(df, required_columns)
        print("✅ CSV columns validated.\n")

        # ----------------------------
        # 5. Validate numeric columns (dose & OD)
        validate_numeric_columns(df, [DOSE_FIELD, OD_FIELD])
        print("✅ Dose and OD columns are numeric.\n")

        # ----------------------------
        # 6. Benchmark and run analysis
        # ----------------------------
        start = timeit.default_timer()
        print("🚀 Running composite score analysis...\n")

        best_timepoints = timepoint_composite_score(
            df=df,
            group_fields=GROUP_FIELDS,
            dose_field=DOSE_FIELD,
            od_field=OD_FIELD,
            time_field=TIME_FIELD,
            standardize=True,
            method="minmax",
        )

        # ----------------------------
        # 7. Display and summarize results
        # ----------------------------
        print("🎯 Best Timepoints (composite score):")
        print(best_timepoints, "\n")

        elapsed = round(timeit.default_timer() - start, 2)
        print(f"⏱️  Program finished in {elapsed} seconds.\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


## EXECUTE
if __name__ == "__main__":
    main()
    # input("Press any key to exit...")
