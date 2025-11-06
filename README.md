# Dose-Response Curve (DRC) Timepoint Selection
## 📦 Repository Contents (Key Files)
- `drc_timepoint_composite_score.py`: Main script for composite scoring timepoints
- `drc_timepoint_selection.ipynb`: old prototype file
- `requirements.txt`: Required Python packages
- `config_*.json`: Example configuration files
- `data/*.csv`: Example datasets
- `drc_concepts.md`, `drc_timepoint_algorithm.md`: Documentation / references


## 🧪 `drc_timepoint_composite_score.py`
Selects the best timepoint for dose-response modeling using **five metrics**:

1. Signal-to-Noise Ratio (SNR)
2. Correlation (Spearman)
3. Coefficient of Variation (CV)
4. Dynamic Range
5. Smoothness (Mean absolute difference)


### 🚀 How to Run
Run the script by passing your JSON configuration file as a command-line argument.
```bash
python drc_timepoint_composite_score.py "Path\to\your\config.json"
```

Example config.json:
```json
{
  "file_path": "data/timepoint_vallo.csv",
  "group_fields": ["Species"],
  "dose_field": "Dose",
  "od_field": "RawOD",
  "time_field": "Time"
}
```
**Explanation of keys:**
- file_path: Path to your CSV file
- group_fields: List of column(s) for grouping (even if single column, it must be in a list)
- dose_field: Column representing numeric dose
- od_field: Column representing raw optical density
- time_field: Column representing timepoints



### ⚠️ Validation Rules
1. JSON config file must be of valid JSON format and contain all required keys.
2. CSV must contain all the columns (exactly matching) you specified in your config file.
3. dose_field and od_field must be numeric.
4. OD values are standardized automatically; do not give pre-transform data as od_field. OD values should be in raw form.



### 📊 Results
The script has been tested on two datasets:
- Performs well on timepoint_vallo.csv ✅
- Shows inconsistent or unexpected results on timepoint_sf.csv ⚠️

Further testing is recommended before general deployment.


**Comparison b.w manually chosen and script suggested time points**

-------------------------------------------------------------------------------------------------
| Dataset             | Condition       | Manual Selected Timepoint | Script Suggested Timepoint|
|---------------------|-----------------|---------------------------|---------------------------|
| timepoint_vallo.csv | -               | 9.5 – 10.5                | 12.97                     |
| timepoint_sf.csv    | SF              | 6:59:35                   | 38.00083                  |
| timepoint_sf.csv    | SFP             | 6:59:35                   | 4.99250                   |
| timepoint_sf.csv    | 20MSynComm      | 32:59:58 OR 32:59:59      | 35.00028                  |
| timepoint_sf.csv    | 20MSynComm+ SF  | 32:59:58                  | 5.99278                   |
| timepoint_sf.csv    | 20MSynComm+ SFP | 32:59:58                  | 5.99306                   |
|  new data come here | -               | 00:00:00                  | 00000                     |
-------------------------------------------------------------------------------------------------


### ⚖️ Future improvements
- more testing needed
- maybe swap SNR and opt for zfactor-like as many of scientific research suggests
- smoothness metric even as of now have really small weight but might be dropped entirely