Each metric contributes to the composite score:
- Signal-to-noise ratio (SNR) → strength of signal relative to noise (higher is better)
- Dose–response correlation → consistency/monotonicity of response across doses (higher is better)
- Coefficient of variation (CV) → replicate reliability (lower is better; inverted in composite score)
- Dynamic range → magnitude of the dose–response effect (higher is better)
- Smoothness (Mean Absolute Difference between successive doses) → informative stepwise changes between doses (moderate-to-high values are preferred; very low indicates a flat, uninformative curve)

## Standardizing OD values
The standardize_od() function transforms raw OD (Optical Density) values onto a consistent scale, allowing fair comparison across conditions, doses, and timepoints.
In this workflow, standardization is optional and applied globally across the dataset.

### Why Global Standardization
The goal is to identify the timepoint where the overall dose–response signal is strongest and most reliable. Global scaling ensures that all OD measurements share a common reference scale, preserving real differences in signal magnitude between groups.

If standardization were done per group (e.g., per Condition, Ratio, or Plate), each group would be rescaled independently, which would:
- Remove meaningful differences in absolute intensity,
- Artificially equalize weak and strong responses, and
- Distort metrics like SNR, dynamic range, or correlation.

Global standardization maintains biological and statistical integrity, ensuring that composite metrics remain comparable across groups and timepoints.

### zscore
Centers data around zero and expresses each value in terms of standard deviations from the global mean:

$$
z=\frac{x−μ}{σ}
$$

### minmax
Rescales all OD values to the [0, 1] range, preserving the data’s original shape:

$$
x_{norm} =\frac{x−x_{min}}{x_{max}-x_{min}}
$$

By default, standardize is False but can be manually set to True within the timepoint_composite_score() function.


## 1. Signal to noise ratio (SNR)
A global SNR (based on maximum–minimum OD differences) is used to quantify how clearly the overall dose–response pattern stands out above measurement variability. 
##### note: for more detailed information on global & local SNR please refer to the document **drc_concepts.md**

### Computation Steps
1. Compute the mean OD for each dose level.
2. Signal = difference between maximum and minimum mean OD values.
3. Noise = average standard deviation of OD across all doses.

$$
SNR = \frac{Max(Mean OD)−Min(Mean OD)}{Average SD}
$$

### Interpretation
A higher SNR indicates a stronger and more distinguishable dose–response effect, while a lower SNR suggests that random variability dominates the signal.


## 2. Dose-response correlation
Spearman correlation measures the consistency of the relationship between dose and OD across all dose levels.

### Computation Steps
1. Compute the mean OD for each dose level.
2. Calculate the Spearman correlation between dose and mean OD values.
3. Take the absolute value of the correlation to focus on the strength, not direction.

$$
Correlation = ∣Spearman(Dose,Mean OD)∣
$$

### Interpretation
A high absolute correlation indicates a clear, monotonic dose–response trend (OD consistently increases or decreases with dose). A low value suggests weak or inconsistent responses across doses.


## 3. Coefficient of variation (CV) for replicates
The CV quantifies the relative variability of replicate measurements at each dose level within a timepoint, providing a measure of data precision.

### Computation Steps
1. For each dose level, compute the mean OD across replicates.
2. Compute the standard deviation (SD) of replicates at that dose.
3. Calculate the coefficient of variation for that dose.

$$
CV=\frac{StandardDeviation}{Mean}
$$

### Interpretation
A low CV indicates stable, consistent replicates, while a high CV indicates noisy or unreliable measurements.


## 4. Dynamic range (ratio of max to min response)
Dynamic range measures the magnitude of the dose–response effect at a given timepoint, capturing how much OD changes across all doses.

### Computation Steps
1. Compute the mean OD for each dose level.
2. Identify the maximum mean OD and minimum mean OD.
3. Calculate the dynamic range.

$$
Dynamic Range=\frac{Max(Mean OD)}{Min(Mean OD)}
$$

### Interpretation
A high dynamic range indicates a pronounced dose–response effect, with OD values changing substantially across doses. A low dynamic range indicates a weak or flat response.
##### note: code handles min == 0 edge case to avoid division by zero.


## 5. Smoothness (Mean Absolute Difference (MAD) between successive doses)
This metric quantifies the smoothness and consistency of the dose–response curve at a given timepoint by measuring the average change between consecutive doses.

### Computation Steps
1. Compute the mean OD for each dose level.
2. Calculate the difference between successive doses:

$$
\Delta_i = OD_{i+1} - OD_i
$$

3. Take the absolute value of each difference.
4. Compute the mean of these absolute differences:

$$
\text{Mean Absolute Difference} = \frac{1}{n-1} \sum_{i=1}^{n-1} |\Delta_i|
$$

Where:  
- $\displaystyle OD_i$ = mean OD at dose $i$  
- $\displaystyle n$ = number of dose levels  
- $\displaystyle |\Delta_i|$ = absolute difference between successive doses

### Interpretation
The MAD captures how much the response changes between successive doses.
- Low MAD → flat curve, minimal change → not informative.
- Moderate MAD → gradual, measurable changes → biologically meaningful and desirable.
- High MAD → abrupt or erratic jumps → may indicate noise; contribution is positive but limited.

Timepoints with moderate-to-high MAD are favored, reflecting informative, stepwise dose–response changes and complementing SNR, correlation, and dynamic range.


## Assigning weights to calculate composite score for best timepoint selection
After computing all metrics for each timepoint, the algorithm combines them into a single composite score to identify the best timepoint for each experimental group. This step ensures that multiple criteria—signal strength, reproducibility, magnitude, and smoothness—are considered simultaneously.

### Computation Steps
1. Assign weights to each metric based on its importance:
- SNR → 0.3
- Correlation → 0.3
- Coefficient of Variation (CV) → -0.2 (negative because lower CV is better)
- Dynamic Range → 0.1
- Smoothness (MAD) → 0.1
2. Normalize each metric to [0,1] via min-max scaling

$$
\text{Metric}_{\text{norm}} = \frac{\text{Metric} - \text{Min(Metric)}}{\text{Max(Metric)} - \text{Min(Metric)}}
$$

3. Invert CV so that lower values contribute positively

$$
\text{Metric}_{\text{norm}} = 1 - \text{Metric}_{\text{norm}}
$$

4. Multiply each normalized metric by its absolute weight and Sum all weighted metrics to get composite score

$$
\text{Composite Score} = \sum (\text{Metric}_{\text{norm}} \times |\text{Weight}|)
$$

5. Select timepoint with highest composite score

### Interpretation & Notes
- SNR: Higher values favor timepoints where the signal clearly stands out from noise.
- Correlation: High absolute correlation ensures strong and consistent dose–response trends.
- CV: Negative weight ensures that low CV (precise replicates) contributes positively.
- Dynamic Range: Higher dynamic range favors timepoints with pronounced biological effects.
- Smoothness (MAD): Moderate-to-high values indicate steady, informative stepwise changes. Very low values reduce contribution, while extremely high values are limited in influence.



## Test results

| Dataset             | Condition       | Manual Selected Timepoint | Script Suggested Timepoint|
|---------------------|-----------------|---------------------------|---------------------------|
| timepoint_vallo.csv | -               | 9.5 – 10.5                | 12.97                     |
| timepoint_sf.csv    | SF              | 6:59:35                   | 5.99278                   |
| timepoint_sf.csv    | SFP             | 6:59:35                   | 5.99306                   |
| timepoint_sf.csv    | 20MSynComm      | 32:59:58 OR 32:59:59      | 32.99944                  |
| timepoint_sf.csv    | 20MSynComm+ SF  | 32:59:58                  | 6.99306                   |
| timepoint_sf.csv    | 20MSynComm+ SFP | 32:59:58                  | 5.99306                   |
|  new data come here | -               | 00:00:00                  | 123456                    |
-------------------------------------------------------------------------------------------------

## Future implementation:
- Add the standardize=True and method="minmax" or "zscore" options as parameters in the JSON config file instead of hardcoding them in main(), making the workflow more dynamic.
- currently tested on two datesets, more testing needed