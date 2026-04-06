For in vitro assays (cell or enzyme assays), following are key terms to understand first.

## Key Terms
| Term               | Variable Type   | Meaning                                           |
|--------------------|-----------------|---------------------------------------------------|
| Concentration/Dose | Independent (x) | Drug/compound amount in well (µM, nM)             |
| Response           | Dependent (y)   | Cell or enzyme activity (%)                       |
| EC₅₀ / IC₅₀        | Parameter       | Concentration for 50% effect (potency)            |
| Emax / Emin        | Parameter       | Max/min achievable response (efficacy)            |
| Hill Slope         | Parameter       | Curve steepness (sensitivity/cooperativity)       |
| Selectivity Index  | Derived ratio   | Safety margin between target and non-target cells |
--------------------------------------------------------------------------------------------

## Concentration and Response
* **Concentration/Dose:** Amount of drug or compound added to the assay. Independent variable (x-axis).  
* **Response:** Biological activity measured from cells or enzymes. Dependent variable (y-axis).


## EC₅₀ / IC₅₀
* **EC₅₀:** Concentration producing 50% of maximal effect. Lower EC₅₀ indicates higher potency.  
* **IC₅₀:** Concentration producing 50% inhibition. Used for inhibitors or blockers.


## Emax / Emin
* **Emax:** Maximum effect the drug can produce (≈EC90).  
* **Emin:** Minimum effect, often baseline or no drug.


## Hill Slope
* Measures steepness of the dose-response curve.  
* Steeper slope → small concentration change produces large response change.


## Selectivity Index (SI)
This measures the safety margin of a drug between different cell types.

```text
SI = IC₅₀ in non-target cells / IC₅₀ in target cells
```

A higher selectivity index means the drug is safer — it affects target cells at much lower concentrations than non-target cells.


## Quick Reference
- **Low EC₅₀/IC₅₀**: Potent (strong effect at low concentration)
- **High EC₅₀/IC₅₀**: Weak (needs more compound for same effect)


## Plate (96‑well) — mental picture
A 96‑well plate has 8 rows (A–H) and 12 columns (1–12). Each small box below is a well; each well can contain a different treatment.
```
Columns →  1    2    3    4   ...   12
Row A   | A1 | A2 | A3 | A4 | ... | A12 |
Row B   | B1 | B2 | B3 | B4 | ... | B12 |
Row C   | C1 | C2 | C3 | C4 | ... | C12 |
 ...
Row H   | H1 | H2 | H3 | H4 | ... | H12 |
```
With these terms and assay setup in mind, we can now examine how to select the optimal time point for dose-response modeling.
The next section introduces five quantitative metrics — SNR, Dose-Response Correlation, CV, Dynamic Range, and Smoothness — that help identify the most reliable and biologically meaningful time points.


## Selecting the Ideal Time Point for Modeling
Before analyzing and modeling the data, it is important to choose an ideal time point where the dose-response relationship is strongest. This can be a tedious task for non-biologists or those without prior experience, as selecting the best time point by visually inspecting the data is not straightforward. The script **drc_timepoint_composite_score.py** helps approximate the most suitable time points for DRC modeling, making this process easier for non-experts.


# Key Concepts for Time Point Selection
1. Signal-to-Noise Ratio (measure of how strong the signal is compared to background noise)
2. Dose-Response Correlation (how well response changes with dose)
3. Coefficient of Variation (variation relative to mean)
4. Dynamic Range (difference between minimum and maximum response)
5. Smoothness (mean absolute difference between successive doses, indicates consistency)

A detailed explanation of the algorithm is provided in **drc_timepoint_algorithm.md**. Before diving into that, we will first review the theoretical concepts behind the five key points listed above.


## 1. Signal-to-Noise Ratio (SNR)
Think of SNR as a measure of how clearly you can “see” the real pattern (signal) in your data amid random variations (noise). In general, SNR quantifies how strong a signal is relative to background noise:

$$
\text{SNR} = \frac{\text{Signal}}{\text{Noise}}
$$

A higher SNR means the signal is clearly distinguishable from noise; a low SNR means the signal is buried in noise.

### Why Noise Is Represented by Standard Deviation
Noise reflects how much your measurements fluctuate randomly. The standard deviation (SD) quantifies that fluctuation, measuring the average spread of your data points around their mean:  

* **Small SD → low noise**: measurements are consistent, signal is clear.  
* **Large SD → high noise**: measurements are unstable, signal is obscured.  

This is why SD is used as a practical and mathematical measure of “noise.”

### Types of SNR in Dose–Response Analysis
Two common approaches to estimating SNR are described below. They differ mainly in how they define the signal.

#### 1.1 Global SNR (Based on Maximum–Minimum Differences)
**Approach:**
1. Compute the mean OD for each dose level.  
2. Signal: difference between the maximum and minimum mean OD across doses.  
3. Noise: average standard deviation of OD values across all dose levels.

$$
\text{SNR} = \frac{\text{Max(Mean OD)} - \text{Min(Mean OD)}}{\text{Average SD}}
$$

**Interpretation:**  
Captures the overall range of the response — how much the highest and lowest mean responses differ relative to background noise.

**Strengths:**
* Simple and intuitive: directly measures total spread of the response curve.  
* Stable summary: uses mean and SD, reducing the impact of random variation.

**Limitations:**
* Overly global: assumes the largest difference (max–min) reflects the true effect, even if caused by outliers or nonlinear behavior.  
* Ignores control: doesn’t consider baseline or dose-specific relationships.  
* Averages noise: assumes all doses have similar variability.

#### 1.2 Local SNR (Relative to Control)
**Approach:**
1. Identify a control dose (typically the lowest or zero dose).  
2. For each dose level: compute the mean OD and calculate the signal as the absolute difference between that dose’s mean OD and the control’s mean OD.  
3. Compute the mean absolute signal across all doses.  
4. Noise: average standard deviation of OD values across doses.

$$
\text{SNR} = \frac{\text{Mean Absolute Signal}}{\text{Average SD}}
$$

**Interpretation:**  
Compares each treatment dose to the control, reflecting how much the response changes relative to baseline.

**Strengths:**
* Baseline-aware: explicitly accounts for the control group, which is biologically meaningful.  
* Dose-sensitive: considers variation at each dose level.  
* More robust: averages differences rather than relying on a single extreme value.

**Limitations:**
* More complex: requires distinguishing control vs. treatment groups.  
* Control-dependent: if the control group is noisy, it may underestimate SNR.


## 2. Dose-Response Correlation
Correlation measures how strongly two variables move together — in this case, how OD changes with dose.  
* **Positive correlation** (close to +1): higher doses consistently produce higher OD.  
* **Negative correlation** (close to –1): higher doses consistently produce lower OD.  
* **No correlation** (around 0): OD changes randomly with dose.  

In short, correlation answers the question:  
**Does OD change consistently as the dose changes?**

### Why Spearman Correlation
Spearman correlation evaluates **rank order** (monotonicity), rather than requiring a strictly linear relationship. This is ideal for biological dose–response data, which are often **curved or sigmoidal** rather than perfectly straight.  

*Works even if the response increases and then plateaus.*

### Why Use the Absolute Value
Taking the absolute value ensures the algorithm focuses on the **strength of the relationship**, ignoring direction. In biological terms, both increases and decreases in OD with dose can be meaningful:  

* **|+1| or |–1| → strong relationship**  
* **0 → no consistent trend**  

**Interpretation:**
* High absolute correlation → OD changes consistently with dose (strong monotonic trend).  
* Low absolute correlation → OD varies unpredictably with dose (weak or inconsistent trend).  

**Strengths:**
* **Monotonic detection:** Works for curved or sigmoidal dose-response curves.  
* **Robust to direction:** Both increasing and decreasing responses are captured.  
* **Simple and intuitive:** Provides a single metric summarizing dose-response consistency.

**Limitations:**
* **Ignores magnitude:** Focuses on order, not the size of response differences.  
* **Sensitive to ties:** Many identical values (common in plateau regions) can reduce correlation.  
* **Cannot capture non-monotonic trends:** If response rises then falls, correlation may underestimate relationship.


## 3. Coefficient of Variation (CV)
The CV measures relative variability, expressing how large the fluctuations are compared to the average signal. It indicates how consistent replicate measurements are relative to their mean:
CV is **dimensionless**, meaning it expresses variability as a percentage or ratio of the mean. This allows us to quantify how noisy or stable the data are at each time point.

$$
\text{CV} = \frac{\text{Standard Deviation}}{\text{Mean}}
$$

```python
cv_mean = (group['std_od'] / group['mean_od'] + 1e-8).mean() # can be effected by outliers 
cv_median = (group['std_od'] / (group['mean_od'] + 1e-8)).median() # takes median so its more robust to the outliers
```

**Approach:**
Each dose typically has multiple replicates. To compute CV:

1. Compute the **mean** and **standard deviation (SD)** for each dose.  
2. Calculate the CV for each dose.  
3. Average the CVs across doses to get an overall measure of replicate stability for the time point.

| Metric | Meaning |
|--------|---------|
| **Mean** | Central response (signal level) |
| **SD** | Uncertainty in the signal (noise) |
| **CV** | Relative size of uncertainty compared to the signal |

**Interpretation:**
* **Low CV → high consistency:** replicates are stable and results are more trustworthy.  
* **High CV → large fluctuations:** replicates are noisy, reducing confidence in the signal.

**Strengths:**
* **Relative measure:** Accounts for variability relative to the signal, making comparisons across time points fair.  
* **Dimensionless:** Works even if the mean signal varies between time points.  
* **Highlights replicate consistency:** Ensures time points with strong, stable measurements are prioritized.

**Limitations:**
* **Sensitive to outliers:** Extreme replicate values can inflate CV.  
* **Ignores direction of changes:** Only quantifies variability, not whether the signal increases or decreases.  
* **Averages may hide local fluctuations:** High variability at specific doses may be masked by averaging across doses.

### Why CV Matters
* A time point can have a strong mean signal (high OD), but if replicates vary wildly (high SD), the result is less reliable.  
* Another time point might have smaller signals but very consistent replicates (low CV), making it more statistically robust.  

By considering CV, we prioritize time points that not only have strong signals but also **reproducible, reliable measurements**.


## 4. Dynamic Range
Dynamic Range measures the spread between the largest and smallest values in a dataset relative to the smallest (baseline) value. It quantifies how much the response changes across doses at a given time point:

$$
Dynamic Range=\frac{Max(Mean OD)}{Min(Mean OD)}
$$

* Max(Mean OD) = the largest mean OD observed across doses.
* Min(Mean OD) = the smallest mean OD observed across doses (baseline).

A large dynamic range indicates that the experimental treatment produces a substantial change in OD, whereas a small dynamic range indicates a weak or flat response.

**Interpretation:**
* **High dynamic range → strong signal:** the response changes substantially across doses.  
* **Low dynamic range → weak signal:** the response changes little across doses.

**Strengths:**
* **Captures signal magnitude:** Measures the absolute size of the dose-response effect, even if variability is low or correlation is high.  
* **Timepoint prioritization:** Allows comparison across time points to select those with the most pronounced response.  
* **Biological relevance:** Highlights time points where the treatment effect is meaningful and detectable.

**Limitations:**
* **Ignores variability:** Does not account for replicate consistency (CV) or noise.  
* **Sensitive to outliers:** A single extreme value can inflate the dynamic range.  
* **Baseline-dependent:** Small baseline values can exaggerate the ratio.

### Why Dynamic Range Matters
Even if a time point has low variability and strong correlation, a small dynamic range may indicate that the response is not biologically significant. By considering dynamic range, the algorithm prioritizes time points where the signal is **both strong and meaningful**, improving the reliability of dose-response modeling.


## 5. Smoothness
Smoothness measures the **consistency of the dose–response curve** by quantifying how much the mean OD changes from one dose to the next. It is calculated as the **mean absolute difference (MAD)** between consecutive doses.

- For a given timepoint, you have a series of mean OD values for each dose (ordered by increasing dose).
- The algorithm computes the absolute differences between each pair of consecutive doses and then averages them.

```python
group["mean_od"] # array of mean OD values for all doses at this timepoint.
np.diff() # computes differences between successive OD values.
np.abs() # converts all differences to positive numbers (ignores direction).
.mean() # takes the average absolute change between doses.
```
So the result is one number per timepoint, representing the average step size between consecutive doses.

**Approach**
1. For a given time point, compute the mean OD at each dose (ordered by increasing dose).  
2. Calculate the absolute difference between each pair of consecutive doses.  
3. Average these differences to obtain a single number per time point:

$$
\text{Smoothness (MAD)} = \frac{1}{n-1} \sum_{i=1}^{n-1} |OD_{i+1} - OD_i|
$$

**Interpretation:** 
* **Very low MAD (~0):** flat curve → little to no informative change → likely not ideal.  
* **Moderate MAD:** measurable, gradual changes between doses → biologically informative.  
* **Very high MAD:** large or irregular jumps → may indicate noise, artifacts, or erratic responses.

A lower MAD indicates a smoother, more gradual curve; a higher MAD indicates larger or more erratic changes between doses.

**Strengths:**
* **Detects irregular curves:** Even if SNR or correlation is high, erratic dose-to-dose jumps are flagged.  
* **Supports reproducibility:** Smooth, gradual changes are easier to model and interpret biologically.  
* **Complements other metrics:** Provides a local, stepwise check alongside SNR (global signal), correlation (monotonicity), and dynamic range.

**Limitations:**
* **Does not capture global signal magnitude:** A very smooth curve can still have small changes (low dynamic range).  
* **Sensitive to extreme jumps:** Outliers between doses can inflate MAD.  
* **Requires ordered doses:** Only meaningful if doses are consistently ordered.

### Why Smoothness Matters
Biological dose–response curves typically change gradually. By including smoothness, the algorithm favors time points where responses are **measurable, consistent, and biologically meaningful**, avoiding time points with erratic or noisy measurements.  

Smoothness is given a **positive but modest weight** in the composite score, ensuring that moderate, informative changes contribute positively without overemphasizing large, erratic jumps. When combined with SNR, correlation, and dynamic range, it helps select the most reliable time points for modeling.
