## DEPENDENCIES
try:
    import sys
    import json
    import timeit
    import logging
    import numpy as np
    import pandas as pd
    from scipy import stats
    from pathlib import Path
    from datetime import datetime
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except ModuleNotFoundError as e:
    missing_module = e.name
    print(f"❌ Required module not found: '{missing_module}'.")
    sys.exit(1)
except ImportError as e:
    # For C-extension or version issues
    print(f"❌ ImportError: {e}")
    print("   This may happen if the package is partially installed or incompatible.")
    sys.exit(1)


## LOGGING SETUP
# log file name setter
def get_default_log_filename(log_name=None):
    script_dir = Path(__file__).resolve().parent
    if not log_name:
        log_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return script_dir / log_name


# Logger setup
def setup_logger(name=__name__, log_file=None, log_to_file=False):
    if log_file is None:
        log_file = get_default_log_filename()

    # Create logger (either creates a new logger or returns an existing one with that name.)
    logger = logging.getLogger(name)
    # This logger will handle all messages from DEBUG and above (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    logger.setLevel(logging.DEBUG)

    # log message formatting for both console and file (timestamp, name, log level, your log message, line number where log was called, filename of the script)
    long_formatter = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s: Line:%(lineno)d: [%(filename)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    short_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (always active) to send logs to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(short_formatter)
    # Only INFO and higher are shown in console, DEBUG is silent
    console_handler.setLevel(logging.INFO)

    # Avoid adding multiple handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        if log_file is None:
            log_file = get_default_log_filename()
        file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        file_handler.setFormatter(long_formatter)
        file_handler.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

    return logger


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
        raise KeyError(f"Missing key(s) in config: {missing}")

    return config


# CSV Reader
def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame with robust error handling."""
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


# Columns matcher
def match_columns(df, required_columns):
    """Matches required columns with those in the DataFrame."""
    df_columns = [col.strip().lower() for col in df.columns]
    missing = [col for col in required_columns if col.lower() not in df_columns]
    if missing:
        raise ValueError(
            "CSV file is missing required column(s) as specified in the config file. \n"
            f"Missing required column(s) from the CSV: {missing}. \n"
        )


# Columns validator
def validate_numeric_columns(df, numeric_columns):
    """Validates that specified columns contain only numeric values.Raises informative errors if non-numeric or NaN values are found."""
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


def standardize_od(df, od_field, method="minmax"):
    """
    Standardize OD values using sklearn scalers (z-score or min-max).
    Parameters
    ----------
    df : pandas.DataFrame. Input dataframe containing OD readings.
    od_field : str. Column name of OD values to standardize.
    method : {'zscore', 'minmax'}, optional (default 'minmax').
    Returns
    -------
    pandas.Series with Standardized OD values.
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
def compute_metrics(df, group_fields, dose_field, time_field, od_field):
    """
    Compute SNR, correlation, CV, dynamic range, smoothness for each group/time.
    Parameters:
        df (pd.DataFrame): Input dataframe
        group_fields (list of str): Columns to group by (e.g., ['Species'])
        dose_field (str): Column name for dose
        time_field (str): Column name for time
        od_field (str): Column name for OD values
    Returns:
        pd.DataFrame: Each row is a group/timepoint with all metrics computed
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
def compute_composite_score(df, weights):
    """Normalizes metrics and applies weights to compute composite score."""
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
def select_optimal_timepoint(df, group_fields):
    """Select optimal timepoint per group."""
    # returns the index of the row with the maximum value in each group.
    idx = df.groupby(group_fields)["composite_score"].idxmax()
    # selects rows by index labels and return that
    return df.loc[idx].reset_index(drop=True)


def main():
    """
    Main entry point for the dose-response analysis program.
    Accepts a JSON config file as a command-line argument and computes
    the optimal timepoint using composite scoring of multiple metrics.
    """
    # Weights dictionary, Order of priority : [SNR, Correlation, CV, Dynamic range, smoothness] as found in research/literature
    weights = {
        "snr": 0.35,
        "correlation": 0.30,
        "cv": -0.20,  # Negative because lower is better and we can invert the normalization results
        "dynamic_range": 0.10,
        "smoothness": 0.05,  # Negative because lower is better
    }

    # Setup logger to log to console only
    logger = setup_logger(log_to_file=False)

    # ----------------------------
    # 1. Parse and validate CLI args
    # ----------------------------
    if len(sys.argv) < 2:
        logger.error(
            "HowToRun: python <drc_timepoint_composite_score.py> <path/to/config_file.json>"
        )
        sys.exit(1)

    config_file = Path(sys.argv[1])

    logger.info(f"🔧 Loading config file: {config_file} \n")

    try:
        # ----------------------------
        # 2. Load and validate config
        # ----------------------------
        config = load_config(config_file)
        logger.info("✅ Success: configuration loaded.\n")

        # Extract values from config
        FILE_PATH_STR = config["file_path"]
        GROUP_FIELDS = config["group_fields"]
        DOSE_FIELD = config["dose_field"]
        OD_FIELD = config["od_field"]
        TIME_FIELD = config["time_field"]
        STANDARDIZE_METHOD = config.get(
            "standardize_method", "minmax"
        )  # default to minmax
        EXPORT_RESULTS = config.get("export_results", False)  # default to False

        # ----------------------------
        # 3. Read CSV file
        # ----------------------------
        logger.info(f"📄 Reading data file: {FILE_PATH_STR}. \n")
        df = read_csv_file(FILE_PATH_STR)

        # ----------------------------
        # 4. Validate required columns
        # ----------------------------
        required_columns = GROUP_FIELDS + [DOSE_FIELD, OD_FIELD, TIME_FIELD]
        match_columns(df, required_columns)
        logger.info(
            "✅ Success: CSV columns match those specified in the JSON config.\n"
        )

        # ----------------------------
        # 5. Validate numeric columns (dose & OD)
        validate_numeric_columns(df, [DOSE_FIELD, OD_FIELD])
        logger.info(
            "✅ Success: 'Dose' and 'OD' columns contain valid numeric data & no missing values.\n"
        )

        # ----------------------------
        # 6. Benchmark and run analysis
        # ----------------------------
        start = timeit.default_timer()
        logger.info("🚀 Starting composite score analysis...\n")

        # 6.1: Standardize OD values
        df[f"{OD_FIELD}_std"] = standardize_od(df, OD_FIELD, method=STANDARDIZE_METHOD)
        logger.info("🔄 OD values standardized using %s method.", STANDARDIZE_METHOD)

        # 6.2: Compute metrics for each group/timepoint, make sure to use standardized OD field
        metrics_df = compute_metrics(
            df, GROUP_FIELDS, DOSE_FIELD, TIME_FIELD, f"{OD_FIELD}_std"
        )
        logger.info("🧮 Metrics computed for all group/timepoints.")

        # 6.3: Normalize metrics to compute composite score
        normalized_metrics_df = compute_composite_score(metrics_df, weights)
        logger.info("📊 Composite scores computed for all group/timepoints. \n")

        # 6.4: Determine optimal timepoint using composite score
        best_timepoint_df = select_optimal_timepoint(
            normalized_metrics_df, GROUP_FIELDS
        )

        # Optional: export to csv
        if EXPORT_RESULTS:
            input_path = Path(FILE_PATH_STR)
            # Default timestamped filename in same folder as input CSV
            output_file = (
                input_path.parent
                / f"{input_path.stem}_composite_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            best_timepoint_df.to_csv(output_file, index=False)
            logger.info(f"💾 Result exported to CSV: {output_file}. \n")
        else:
            logger.info(
                "📌 Export to CSV not requested. Results available only in console. \n"
            )

        logger.info("🔔 Composite score analysis completed.\n")

        # ----------------------------
        # 7. Display and summarize results
        # ----------------------------
        logger.info(
            "🎯 Result: Best Timepoints (composite score):\n%s\n", best_timepoint_df
        )

        elapsed = round(timeit.default_timer() - start, 2)
        logger.info(f"🕐 Program finished in {elapsed} seconds.\n")

    except Exception as e:
        logger.error(f"\n❌ Error: %s", e)
        sys.exit(1)


## EXECUTE
if __name__ == "__main__":
    main()
    # input("Press any key to exit...")
