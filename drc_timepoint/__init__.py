"""
drc_timepoint — Dose-response curve timepoint selection package.

Public API:
    run_analysis_from_config   — full pipeline from a config dict
    load_config                — load + validate a JSON config file
    read_csv_file              — read + validate a CSV file
    standardize_od             — standardize OD values (zscore / minmax)
    compute_metrics            — compute SNR, correlation, CV, dynamic range, smoothness
    compute_composite_score    — normalize metrics and compute weighted composite score
    select_optimal_timepoint   — pick the best timepoint per group based on composite score
"""

from .io import load_config, read_csv_file
from .validation import match_columns, validate_numeric_columns
from .analysis import (
    standardize_od,
    compute_metrics,
    compute_composite_score,
    select_optimal_timepoint,
)
from .runner import run_analysis_from_config

__all__ = [
    "load_config",
    "read_csv_file",
    "match_columns",
    "validate_numeric_columns",
    "standardize_od",
    "compute_metrics",
    "compute_composite_score",
    "select_optimal_timepoint",
    "run_analysis_from_config",
]
