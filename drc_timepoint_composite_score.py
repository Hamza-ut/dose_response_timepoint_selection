## DEPENDENCIES
import json
import timeit
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


## FUNCTIONS
# JSON config file loader
def load_config(config_file):
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
        return config
    except Exception as e:
        raise ValueError(f"Json config file error: {e}")


# Valid file check
def read_csv_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    if file_path.suffix.lower() != ".csv":
        raise ValueError("Input can be only CSV file")

    return pd.read_csv(file_path)


# Valid columns check
def validate_columns(df, required_columns):
    """Validates that all required columns are present in the DataFrame."""
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"The input CSV must contain the following columns: {required_columns}"
        )


# Standardize OD values
def standardize_od(df, od_field, method):
    """Normalize Raw OD values using minmax or zscore"""
    if method == "zscore":
        return (df[od_field] - df[od_field].mean()) / df[od_field].std()
    elif method == "minmax":
        return (df[od_field] - df[od_field].min()) / (
            df[od_field].max() - df[od_field].min()
        )
    else:
        raise ValueError("Invalid method for standardization.")


# Best timepoint using composite scores
def timepoint_composite_score(
    df,
    group_fields,
    dose_field,
    od_field,
    time_field,
    standardize=False,
    method="minmax",
):
    """
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing experimental data.
        group_fields (list of str): The column names used for grouping (e.g., ['Condition', 'Ratio', 'Plate']).
        dose_field (str): The column name representing the dose values.
        od_field (str): The column name representing the observed OD values.
        time_field (str): The column name representing the time points.
    Returns:
    pd.DataFrame: A dataframe containing the best timepoint for each group, along with calculated metrics and the composite score.
    """
    # Standardize OD values if requested
    if standardize:
        df[od_field] = standardize_od(df, od_field, method)

    # Calculate mean OD for each group, dose, and time
    mean_df = (
        df.groupby(group_fields + [dose_field, time_field])[od_field]
        .agg([("mean_od", "mean"), ("std_od", "std"), ("count", "count")])
        .reset_index()
    )

    # Calculate metrics for each timepoint
    results = []
    for name, group in mean_df.groupby(group_fields + [time_field]):
        # Extract group information
        group_info = dict(zip(group_fields + [time_field], name))

        # 1. Signal-to-noise ratio (mean difference between max and min dose / average std)
        max_signal = group["mean_od"].max() - group["mean_od"].min()
        avg_noise = group["std_od"].mean()
        snr_min_max = max_signal / avg_noise if avg_noise != 0 else 0

        # 2. Dose-response correlation (Spearman correlation between dose and response), take absolute because we are interested in strength not the direction of relationship
        correlation_abs = abs(stats.spearmanr(group[dose_field], group["mean_od"])[0])

        # 3. Coefficient of variation for replicates
        cv_median = (group["std_od"] / (group["mean_od"] + 1e-8)).mean()

        # 4. Dynamic range (ratio of max to min response)
        dynamic_range = (
            group["mean_od"].max() / group["mean_od"].min()
            if group["mean_od"].min() != 0
            else 0
        )

        # 5. Mean absolute difference between successive doses
        dose_response_smoothness = np.abs(np.diff(group["mean_od"])).mean()

        results.append(
            {
                **group_info,
                "snr": snr_min_max,
                "correlation": correlation_abs,
                "coeff_of_variation": cv_median,
                "dynamic_range": dynamic_range,
                "smoothness": dose_response_smoothness,
            }
        )

    results_df = pd.DataFrame(results)

    # Calculate composite score (you can adjust weights based on importance)
    weights = {
        "snr": 0.3,
        "correlation": 0.3,
        "coeff_of_variation": -0.2,  # Negative because lower is better and we can invert the normalization results
        "dynamic_range": 0.1,
        "smoothness": 0.1,
    }

    for metric, weight in weights.items():
        # Normalize the metrics using min max scaling
        results_df[f"{metric}_norm"] = (
            results_df[metric] - results_df[metric].min()
        ) / (results_df[metric].max() - results_df[metric].min())
        # for flipping the scale since for coeff of variation low values replicates are better and high value replicates are worse
        # so after inversion high normalized values (close to 1) become low (close to 0) and low normalized values (close to 0) become high (close to 1).
        if weight < 0:
            results_df[f"{metric}_norm"] = 1 - results_df[f"{metric}_norm"]

    # Calculate weighted composite score
    results_df["composite_score"] = sum(
        results_df[f"{metric}_norm"] * abs(weight) for metric, weight in weights.items()
    )

    # Select best timepoint for each group based on composite score
    best_timepoints_df = results_df.loc[
        results_df.groupby(group_fields)["composite_score"].idxmax()
    ]

    # Select columns to keep dynamically
    columns_to_keep = (
        group_fields
        + [time_field]
        + [
            "snr",
            "correlation",
            "coeff_of_variation",
            "dynamic_range",
            "smoothness",
            "composite_score",
        ]
    )
    best_timepoints_df = best_timepoints_df[columns_to_keep]

    return best_timepoints_df


## MAIN
def main():
    # Ask the user to input the path to the config file
    config_file = input(
        "Please enter the path to your configuration file (e.g., 'config_sf.json'): "
    )
    try:
        # Load the selected config file
        config = load_config(config_file)

        # Extract values from the loaded config
        FILE_PATH_RAW = config["file_path_raw"]
        GROUP_FIELDS = config["group_fields"]
        DOSE_FIELD = config["dose_field"]
        OD_FIELD = config["od_field"]
        TIME_FIELD = config["time_field"]

        # Print the loaded configuration
        print("Config loaded:", config)

        # Read the CSV to df using the customized function
        file_path = Path(FILE_PATH_RAW)
        df = read_csv_file(file_path)

        # Validate required columns
        required_columns = GROUP_FIELDS + [DOSE_FIELD, OD_FIELD, TIME_FIELD]
        validate_columns(df, required_columns)

        # benchmark start time
        start = timeit.default_timer()

        # run the function to find best timepoint
        timepoint_using_composite_score = timepoint_composite_score(
            df,
            GROUP_FIELDS,
            DOSE_FIELD,
            OD_FIELD,
            TIME_FIELD,
            standardize=True,
            method="minmax",
        )
        print(timepoint_using_composite_score)

        # Save the results to a CSV file
        # output_file = file_path.stem + "_timepoint_composite_score.csv"
        # timepoint_using_composite_score.to_csv(output_file, index=False)

        # Time consumed
        end = timeit.default_timer()
        print("Program Finished in {} seconds".format(round((end - start), 2)))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


## EXECUTE
if __name__ == "__main__":
    main()
    # input("Press any key to exit...")
