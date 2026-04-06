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
- Order of priority according to various research articles etc is as follow: [SNR, Correlation, CV, Dynamic range, smoothness]

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



--------------------------------
# Additional read (Explanation of compute_metrics function)
The following section explains the internal mechanics and rationale behind the **compute_metrics()** function — the core logic that helps in identifying optimal timepoint.


### Initial grouping
The first step is to aggregate the raw OD data for all replicates. we combine all replicates (usually from multiple plates or wells).

We group the dataframe by: group_fields + dose_field + time_field
This ensures that for each unique combination of:
- Experimental group (e.g., Species, Condition, etc.)
- Dose
- Timepoint

For each of these combinations, we calculate:
- mean_od → the average OD value across replicates
- std_od → standard deviation of OD across replicates
- count → number of replicates (useful to identify groups with only a single measurement, where std_od cannot be calculated)

### Grouping by Timepoint
Once replicate-level aggregation is complete, we regroup the data by:
- group_fields + time_field
This allows the function to evaluate each timepoint independently for every experimental group, considering how the OD values vary across doses within that timepoint.
Essentially, we’re now comparing dose–response behavior over time to identify the most optimal timepoint.

Now for each of the group

#### Metric 1: SNR
The reason is that SNR (signal-to-noise ratio) is calculated for each timepoint across all doses.
- Max(Mean OD) → maximum mean OD across all doses at this timepoint
- Min(Mean OD) → minimum mean OD across all doses at this timepoint
- Average SD → mean of std_od across all doses at this timepoint

This metric tells us how well the dose-response separates the signal from noise at each timepoint.

#### Metric 2: Correlation
At each timepoint, we want to know: Do OD values change monotonically with dose? 
Monotonically means consistently, either always increasing or always decreasing, with no reversal. That’s exactly what Spearman’s rank correlation coefficient (ρ) measures. We take absolute value because we only care about the strength of the dose-response relationship, not whether OD increases or decreases with dose.

#### Metric 3: Coefficient of Variation
It tells you how large the standard deviation is relative to the mean — basically, how noisy your data is compared to its signal strength.

The function takes the average of all CVs across doses for each timepoint. So, it summarizes how variable (noisy) the dose-response measurements are, overall, at that timepoint. 
- Low CV means replicates are consistent and there is less noise. 
- High CV means replicates vary a lot and there is high noise.

#### Metric 4: Dynamic range
Dynamic range measures how big the signal spread is between the lowest and highest mean OD values across all doses — at a given timepoint.
Dynamic Range (DR) is simply: max(Mean OD)−min(Mean OD). In other words, we are measuring how far apart the highest and lowest mean responses are at that timepoint.

Typically:
- At early timepoints, cells or reactions may not have responded much → small dynamic range.
- At late timepoints, the response might plateau or saturate → again small dynamic range.
- Somewhere in between, you often find a peak dynamic range — that’s usually the optimal timepoint for DRC modelling.

That’s the point where the response is strong enough to differentiate doses, But not yet saturated, and variability (noise) is still manageable.

#### Metric 5: Smoothness
The smoothness metric captures how consistent and gradual the dose–response curve is.
- It computes the difference between consecutive mean OD values (based on dose order).
- Takes the absolute value of each change — ignoring direction (up or down).
- Averages those differences to get the average magnitude of change between doses.

Smaller smoothness values mean a cleaner, more biologically meaningful dose-response curve. It’s more of a sanity check that's why this metric has the lowest weight. 


### Implementation notes:
> group_info is a dictionary containing the group identifiers (group_fields + time_field)
**group_info unpacks the dictionary so it can be merged with the calculated SNR into a single dictionary row. All SNR results are appended to a list and then converted into a dataframe for further processing

> If count = 1, standard deviation (std_od) is NaN. This is expected because you cannot calculate SD with a single value.

> For those rows where standard deviation was NaN, the CV would also be NaN. This is also expected.




