# Dose-Response Curve (DRC) Timepoint Selection
This repository contains tool to select the best timepoint for dose-response curve (DRC) modeling.  

---

### 1. `drc_timepoint_selection.ipynb` (Notebook – For Testing only)
This notebook contains initial experiments for identifying the optimal timepoint for DRC modeling.  
- **Test 1:** Uses standardized raw OD/response values and finds the best timepoint based on the mean absolute difference (variance) between response values.  
- **Test 2:** Uses signal-to-noise ratio (SNR) to identify the best timepoint, assuming the presence of control or minimum dose values.  

> Note: The Jupyter notebook is a simple initial test and not recommended for production use.
> The main utility is provided by the Python script below.

---

### 2. `drc_timepoint_composite_score.py` (Main Script)
This Python script implements a **composite scoring method** to select the best timepoint for DRC modeling using five key metrics:

1. **Signal-to-Noise Ratio (SNR)** – Based on differences between maximum and minimum OD values.  
2. **Correlation** – Spearman correlation is used as a proxy to evaluate the strength of the dose-response relationship.  
3. **Coefficient of Variation (CV)** – Low CV indicates consistent replicates with minimal variability relative to the mean.  
4. **Dynamic Range** – Reflects the assay’s ability to differentiate between the lowest and highest doses.  
5. **Dose Response Smoothness** – Measures how smoothly the response changes with increasing dose, avoiding abrupt fluctuations.

---

### How to run:
run the .py file, the script will ask for config.json as input. your json config file must have the following:
```json
{
    "file_path_raw": "full file path to your csv file",
    "group_fields": "List containing name of columns which will be used for grouping",
    "dose_field": "Str input of column name for Dose",
    "od_field": "Str input of column name for Response/Optical density",
    "time_field": "Str input of column name for time field"
  }
```

> Note: Sample config.json files are included in the repository for reference.