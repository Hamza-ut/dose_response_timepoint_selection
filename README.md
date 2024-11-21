# drc_timepoint_selection.ipynb
## using mean abs diff and SNR ratios to select best timepoint for drc modelling
#### Test1: uses standardized raw OD/response and find best timepoint using mean diff (variance) b.w response values
#### Test2: uses signal to noise ratio to identify best timepoint. It also assumes a control dose (minimum dose)


# drc_timepoint_composite_score.py
## Uses composite score of the following five metrics to select best timepoint for drc modelling
#### 1. SIGNAL TO NOISE RATIO (SNR Based on Maximum and Minimum OD Differences)
#### 2. Correlation (Spearman correlation is used as a proxy to identify the strength of the dose-response relationship)
#### 3. Coefficient of variation (Low CV indicates that the replicates are consistent and have low variability relative to their mean, so those are given higher weightages for scoring)
#### 4. Dynamic range (dynamic range reflects the ability of the assay to differentiate between the lowest and highest doses in the dose-response curve)
#### 5. Dose response smoothness (Measure of how smoothly the responses (e.g., mean OD values) change as the dose increases. It is particularly important for ensuring that the dose-response curve has a consistent, logical trend without abrupt jumps or fluctuations)

### How to run:
run the .py file, the script will ask for json file.json (samples included)

To run the script succesfully, you json config file must have the following:
{
    "file_path_raw": full file path to your csv file,
    "group_fields": List containing name of columns which will be used for grouping,
    "dose_field": Str input of column name for Dose,
    "od_field": Str input of column name for Response/Optical density,
    "time_field": Str input of column name for time field
  }