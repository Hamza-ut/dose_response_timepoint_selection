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
├── data/                               # Example datasets
├── config_sf.json                      # Example config (SF dataset)
├── config_vallo.json                   # Example config (Vallo dataset)
├── code_testing.ipynb                  # Exploratory notebook
├── algorithm_explanation.md            # Algorithm documentation
├── key_concepts.md                     # Background / concepts
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
  "dose_field": "Dose",
  "od_field": "RawOD",
  "time_field": "Time",
  "standardize_method": "minmax",
  "export_results": true
}
```

| Key | Required | Description |
|-----|----------|-------------|
| `file_path` | ✅ | Path to the CSV file |
| `group_fields` | ✅ | Column(s) to group by — must be a list, even for a single column |
| `dose_field` | ✅ | Column with numeric dose/concentration values |
| `od_field` | ✅ | Column with raw OD measurements |
| `time_field` | ✅ | Column with measurement timepoints |
| `standardize_method` | ❌ | `"minmax"` (default) or `"zscore"` |
| `export_results` | ❌ | `true` saves results as CSV next to the input file; `false` prints to console only |

---

## 🐍 Import Usage

The `drc_timepoint` folder is also a importable Python package. You can use it directly in scripts or notebooks:

```python
# Full pipeline from a config dict
from drc_timepoint import run_analysis_from_config, load_config

config = load_config("config_vallo.json")
result_df = run_analysis_from_config(config)

# Or call individual functions
from drc_timepoint import compute_metrics, standardize_od
```

---

## 🧪 Scoring Metrics

Five metrics are computed per timepoint per group, normalized, and combined into a weighted composite score:

| Metric | Weight | Direction | Description |
|--------|--------|-----------|-------------|
| SNR (Signal-to-Noise Ratio) | 0.35 | higher = better | Max signal range divided by mean replicate noise |
| Spearman Correlation | 0.30 | higher = better | Monotonicity of dose–response relationship |
| CV (Coefficient of Variation) | 0.20 | lower = better | Replicate consistency |
| Dynamic Range | 0.10 | higher = better | Ratio of max to min mean OD |
| Smoothness | 0.05 | closer to optimal = better | Consistency of step size across doses |

The timepoint with the highest composite score per group is selected.

---

## ⚠️ Validation Rules

1. Config must be valid JSON and contain all mandatory keys.
2. CSV must contain all columns specified in the config (exact name match).
3. `dose_field` and `od_field` must be numeric with no missing values.
4. OD values should be raw — standardization is applied automatically.
5. `standardize_method` defaults to `"minmax"` if omitted.
6. `export_results` must be a boolean (`true` / `false`).

---

## 📊 Results

Tested on two datasets:

| Dataset | Condition | Manual Timepoint | Script Timepoint | Match |
|---------|-----------|-----------------|-----------------|-------|
| timepoint_vallo.csv | — | 9.5 – 10.5 | 12.97 | ⚠️ close |
| timepoint_sf.csv | SF | 6:59:35 | 38.00083 | ❌ |
| timepoint_sf.csv | SFP | 6:59:35 | 4.99250 | ⚠️ close |
| timepoint_sf.csv | 20MSynComm | 32:59:58 or 32:59:59 | 35.00028 | ⚠️ close |
| timepoint_sf.csv | 20MSynComm + SF | 32:59:58 | 5.99278 | ❌ |
| timepoint_sf.csv | 20MSynComm + SFP | 32:59:58 | 5.99306 | ❌ |

- Performs well on `timepoint_vallo.csv` ✅
- Shows inconsistent results on `timepoint_sf.csv` ❌ — further testing needed before general use.

---

## 🔭 Future Improvements

- More testing across datasets
- Consider replacing SNR with a z-factor-like metric (better supported in literature)
- Smoothness metric currently has very low weight — may be dropped entirely
- Investigate poor performance on `timepoint_sf.csv`
