# Dose-Response Curve (DRC) Timepoint Selection

Selects the optimal measurement timepoint for dose-response modeling using a composite scoring approach across five metrics.

---

## 📁 Repository Structure

```
dose_response_timepoint_selection/
├── drc_timepoint/                      # Core package
│   ├── __init__.py
│   ├── logging_utils.py                # Logger setup
│   ├── io.py                           # Config + CSV loading
│   ├── validation.py                   # Column and type validation
│   ├── analysis.py                     # Metrics + scoring functions
│   └── runner.py                       # Pipeline orchestrator
├── drc_timepoint_composite_score.py    # CLI entry point
├── data/                               # Example datasets + their configs (<dataset>_config.json)
├── tests/                              # Unit + integration tests
├── code_testing.ipynb                  # Exploratory notebook
├── algorithm_explanation.md            # Algorithm documentation
├── key_concepts.md                     # Background / concepts
├── dataset_validation_tracker.xlsx     # Dataset tracker: sources, script vs. expert timepoints
└── requirements.txt
```

---

## 🚀 How to Run (CLI)

```bash
python drc_timepoint_composite_score.py "path/to/config.json"
```

---

## ⚙️ Config File

```json
{
  "file_path": "data/timepoint_vallo.csv",
  "group_fields": ["Species"],
  "dose_field": "uM",
  "od_field": "RawOD",
  "time_field": "Time_h",
  "top_n": 3,
  "export_results": false,
  "export_dir": ""
}
```

| Key            | Required | Default | Description                                                  |
| -------------- | -------- | ------- | ------------------------------------------------------------ |
| file_path      | ✅       | -       | Path to the input CSV file.                                  |
| group_fields   | ✅       | -       | Column(s) to group by (e.g., `["Species"]`). Must be a list. |
| dose_field     | ✅       | -       | Column containing numeric dose/concentration values.         |
| od_field       | ✅       | -       | Column containing raw OD measurements.                       |
| time_field     | ✅       | -       | Column containing measurement timepoints.                    |
| top_n          | ❌       | 3       | Number of top-ranking timepoints to return per group.        |
| export_results | ❌       | false   | If true, saves the final ranking as a CSV file named `<input_filename>_result.csv`. |
| export_dir     | ❌       | `""`    | Directory where the result CSV is written. Can be relative or an absolute path anywhere on disk. If empty or omitted, the file is saved next to the input file. The directory is created automatically if it does not exist. |

---

## ⚠️ Validation Rules

1. Numeric Integrity: dose_field, od_field, and time_field must contain numeric data.
2. Automated Standardization: OD values are standardized using Group-Wise Min-Max scaling. This ensures slow-growing species are evaluated relative to their own growth potential, not global maximums.
3. Time Bucketing: Measurements are rounded to 1 decimal place (precision 1) before analysis to collapse machine drift (seconds/minutes) into meaningful "measurement windows."
4. Window Reporting: The "Ideal Time Window" in the output represents the $\pm 0.05h$ range around the standardized timepoint.
5. Replicate Analysis: To calculate SNR and CV, the script requires multiple measurements per dose. Ensure your group_fields do not include the "Plate" or "Well ID" columns, or the noise-calculation math will have no replicates to compare.

---

## 🐍 Import Usage

The `drc_timepoint` folder is also a importable Python package. You can use it directly in scripts or notebooks:

```python
# Full pipeline from a config dict
from drc_timepoint import run_analysis_from_config, load_config

config = load_config("data/timepoint_vallo_config.json")
result_df = run_analysis_from_config(config)

# Or call individual functions
from drc_timepoint import compute_metrics, standardize_od
```

---

## 🧪 Scoring Metrics

Five metrics are computed for each timepoint per group, normalized, and combined into a weighted composite score:

- SNR (0.30) & Correlation (0.30): These are "Quality of Curve" drivers.
- CV (0.25): If replicates don't agree, the score drops fast.
- Smoothness (0.10): Prevents "jagged" curves where one dose is an outlier.
- Dynamic Range (0.05): A tie-breaker that favors curves with a larger vertical drop.

The Top N timepoints with the highest composite scores per group are selected and returned.

---

## 📊 Results

To assess the accuracy of the tool, for each dataset/group, the timepoint window suggested by the script is compared against a timepoint window determined independently by a domain expert (where available):

| Dataset                          | Group            | Script suggested | Expert suggested    | Verdict                   |
| -------------------------------- | ---------------- | ---------------- | ------------------- | ------------------------- |
| timepoint_vallo.csv              | S. flexneri      | 12.95 – 13.05    | 9.5 – 10.5          | ❌ Significant deviation  |
| timepoint_sf.csv                 | SF               | 4.95 – 6.05      | 6:59:35             | ⚠️ Close                  |
| timepoint_sf.csv                 | SFP              | 4.95 – 6.05      | 6:59:35             | ⚠️ Close                  |
| timepoint_sf.csv                 | 20MSynComm       | 32.95 – 33.05    | 32:59:58 / 32:59:59 | ✅ Match                  |
| timepoint_sf.csv                 | 20MSynComm + SF  | 30.95 – 31.05    | 32:59:58            | ⚠️ Close                  |
| timepoint_sf.csv                 | 20MSynComm + SFP | 30.95 – 31.05    | 32:59:58            | ⚠️ Close                  |
| custom_growth_2-FMA_toxicity.csv | KT2440           | 16.95 – 17.05    | —                   | ⏳ Pending expert opinion |

Full dataset sources, download links, comments, and the underlying script-vs-expert comparisons are tracked in [dataset_validation_tracker.xlsx](dataset_validation_tracker.xlsx).

---

## 🔭 Future Improvements

- More testing across datasets
- Replacing SNR with a z-factor-like metric (better supported in literature)
- Smoothness metric currently has very low weight — may be dropped entirely
