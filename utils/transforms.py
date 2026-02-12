"""Data transformation utilities for clinical research analysis."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def apply_log_transform(df: pd.DataFrame, column: str, base: str = "natural") -> Tuple[pd.DataFrame, str]:
    """Apply log transformation to a column.

    Args:
        df: DataFrame
        column: Column to transform
        base: 'natural', 'log10', or 'log2'

    Returns:
        Transformed DataFrame and new column name
    """
    new_col = f"{column}_log" if base == "natural" else f"{column}_{base}"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")

    # Handle zero and negative values
    min_val = series[series > 0].min() if (series > 0).any() else 1
    series = series.clip(lower=min_val)

    if base == "natural":
        result[new_col] = np.log(series)
    elif base == "log10":
        result[new_col] = np.log10(series)
    elif base == "log2":
        result[new_col] = np.log2(series)

    return result, new_col


def apply_sqrt_transform(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """Apply square root transformation to a column."""
    new_col = f"{column}_sqrt"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")
    series = series.clip(lower=0)
    result[new_col] = np.sqrt(series)

    return result, new_col


def apply_zscore_transform(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """Apply z-score standardization to a column."""
    new_col = f"{column}_z"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")
    mean = series.mean()
    std = series.std()

    if std > 0:
        result[new_col] = (series - mean) / std
    else:
        result[new_col] = 0

    return result, new_col


def categorize_continuous(
    df: pd.DataFrame,
    column: str,
    method: str = "quantiles",
    n_categories: int = 4,
    custom_bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Categorize a continuous variable.

    Args:
        df: DataFrame
        column: Column to categorize
        method: 'quantiles', 'equal_width', or 'custom'
        n_categories: Number of categories for quantiles/equal_width
        custom_bins: Custom bin edges (for method='custom')
        labels: Custom labels for categories

    Returns:
        DataFrame with new column and new column name
    """
    new_col = f"{column}_cat"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")

    if method == "quantiles":
        result[new_col] = pd.qcut(series, q=n_categories, labels=labels, duplicates="drop")
    elif method == "equal_width":
        result[new_col] = pd.cut(series, bins=n_categories, labels=labels)
    elif method == "custom" and custom_bins is not None:
        result[new_col] = pd.cut(series, bins=custom_bins, labels=labels, include_lowest=True)

    return result, new_col


def dichotomize(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    labels: Optional[Tuple[str, str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Dichotomize a continuous variable at a threshold.

    Args:
        df: DataFrame
        column: Column to dichotomize
        threshold: Cutpoint value
        labels: Labels for (below, above) threshold

    Returns:
        DataFrame with new column and new column name
    """
    new_col = f"{column}_bin"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")

    if labels:
        result[new_col] = np.where(series >= threshold, labels[1], labels[0])
    else:
        result[new_col] = (series >= threshold).astype(int)

    return result, new_col


def combine_categories(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    """Combine/recode categories in a variable.

    Args:
        df: DataFrame
        column: Column to recode
        mapping: Dictionary mapping old values to new values

    Returns:
        DataFrame with new column and new column name
    """
    new_col = f"{column}_recoded"
    result = df.copy()

    result[new_col] = result[column].replace(mapping)

    return result, new_col


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.Series:
    """Detect outliers in a numeric column.

    Args:
        df: DataFrame
        column: Column to check
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier (default 1.5) or z-score threshold (default 3)

    Returns:
        Boolean Series indicating outliers
    """
    series = pd.to_numeric(df[column], errors="coerce")

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)
    elif method == "zscore":
        z = (series - series.mean()) / series.std()
        return np.abs(z) > threshold

    return pd.Series(False, index=df.index)


def winsorize(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> Tuple[pd.DataFrame, str]:
    """Winsorize a column (cap extreme values at percentiles).

    Args:
        df: DataFrame
        column: Column to winsorize
        lower_percentile: Lower percentile (default 1%)
        upper_percentile: Upper percentile (default 99%)

    Returns:
        DataFrame with new column and new column name
    """
    new_col = f"{column}_wins"
    result = df.copy()

    series = pd.to_numeric(result[column], errors="coerce")
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)

    result[new_col] = series.clip(lower=lower, upper=upper)

    return result, new_col


def create_interaction(
    df: pd.DataFrame,
    col1: str,
    col2: str,
) -> Tuple[pd.DataFrame, str]:
    """Create an interaction term between two variables.

    Args:
        df: DataFrame
        col1: First column
        col2: Second column

    Returns:
        DataFrame with new column and new column name
    """
    new_col = f"{col1}_x_{col2}"
    result = df.copy()

    # Check if both are numeric
    s1 = df[col1]
    s2 = df[col2]

    if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        result[new_col] = s1 * s2
    else:
        # For categorical, create combined levels
        result[new_col] = s1.astype(str) + "_" + s2.astype(str)

    return result, new_col


def get_transformation_summary(
    original: pd.Series,
    transformed: pd.Series,
) -> Dict[str, Any]:
    """Get summary statistics comparing original and transformed values."""
    orig_numeric = pd.to_numeric(original, errors="coerce")
    trans_numeric = pd.to_numeric(transformed, errors="coerce")

    return {
        "original_mean": orig_numeric.mean(),
        "original_std": orig_numeric.std(),
        "original_min": orig_numeric.min(),
        "original_max": orig_numeric.max(),
        "original_skew": orig_numeric.skew(),
        "transformed_mean": trans_numeric.mean(),
        "transformed_std": trans_numeric.std(),
        "transformed_min": trans_numeric.min(),
        "transformed_max": trans_numeric.max(),
        "transformed_skew": trans_numeric.skew(),
    }
