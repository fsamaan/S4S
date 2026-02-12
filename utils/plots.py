"""Plotting utilities for clinical research analysis."""

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False


def create_forest_plot(
    results_df: pd.DataFrame,
    effect_col: str = "OR",
    ci_lower_col: str = "CI Lower",
    ci_upper_col: str = "CI Upper",
    label_col: str = "Variable",
    title: str = "Forest Plot",
    xlabel: str = "Effect Size",
    null_value: float = 1.0,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    exclude_intercept: bool = True,
) -> Optional[plt.Figure]:
    """Create a forest plot from regression results.

    Args:
        results_df: DataFrame with effect estimates and confidence intervals
        effect_col: Column name for effect estimates (OR, HR, or Beta)
        ci_lower_col: Column name for CI lower bound
        ci_upper_col: Column name for CI upper bound
        label_col: Column name for variable labels
        title: Plot title
        xlabel: X-axis label
        null_value: Reference line value (1 for OR/HR, 0 for Beta)
        log_scale: Use log scale for x-axis (True for OR/HR)
        figsize: Figure size
        exclude_intercept: Exclude intercept from plot

    Returns:
        matplotlib Figure or None if no valid data
    """
    df = results_df.copy()

    # Filter out intercept if requested
    if exclude_intercept and label_col in df.columns:
        df = df[~df[label_col].str.lower().str.contains("intercept", na=False)]

    # Check required columns exist
    required = [effect_col, ci_lower_col, ci_upper_col, label_col]
    if not all(col in df.columns for col in required):
        return None

    # Remove rows with missing values
    df = df.dropna(subset=[effect_col, ci_lower_col, ci_upper_col])

    if df.empty:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each estimate
    n_vars = len(df)
    y_positions = np.arange(n_vars)

    effects = df[effect_col].values
    ci_lower = df[ci_lower_col].values
    ci_upper = df[ci_upper_col].values
    labels = df[label_col].values

    # Calculate errors for errorbar
    xerr_lower = effects - ci_lower
    xerr_upper = ci_upper - effects

    # Plot points and error bars
    ax.errorbar(
        effects,
        y_positions,
        xerr=[xerr_lower, xerr_upper],
        fmt="o",
        color="#4B9EFE",
        ecolor="#64748b",
        capsize=4,
        capthick=1.5,
        markersize=8,
        elinewidth=1.5,
    )

    # Add reference line
    ax.axvline(x=null_value, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.7)

    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    # Configure axes
    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add gridlines
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Invert y-axis so first variable is at top
    ax.invert_yaxis()

    # Add effect size annotations on right side
    x_max = ax.get_xlim()[1]
    for i, (eff, lo, hi) in enumerate(zip(effects, ci_lower, ci_upper)):
        text = f"{eff:.2f} ({lo:.2f}-{hi:.2f})"
        ax.annotate(
            text,
            xy=(x_max, y_positions[i]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#374151",
        )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    return fig


def create_km_plot(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    time_event_cols: Dict[str, Dict[str, str]],
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[plt.Figure]:
    """Create Kaplan-Meier survival curve.

    Args:
        df: DataFrame
        exposure: Exposure/grouping variable
        outcome: Outcome name
        time_event_cols: Mapping of outcome to time/event columns
        figsize: Figure size

    Returns:
        matplotlib Figure or None if not available
    """
    if not LIFELINES_AVAILABLE:
        return None

    mapping = time_event_cols.get(outcome, {})
    time_col = mapping.get("time", outcome + "_time")
    event_col = mapping.get("event", outcome + "_event")

    if time_col not in df.columns or event_col not in df.columns:
        return None

    data = df[[exposure, time_col, event_col]].dropna()
    if data.empty:
        return None

    # Calculate log-rank p-value
    interval = mapping.get("interval")
    p_val = None
    if interval and len(interval) == 2:
        start_t, end_t = interval
        if end_t > start_t:
            data_interval = data[(data[time_col] >= start_t) & (data[time_col] <= end_t)]
            if data_interval[exposure].nunique() == 2:
                groups = data_interval[exposure].unique().tolist()
                g1 = data_interval[data_interval[exposure] == groups[0]]
                g2 = data_interval[data_interval[exposure] == groups[1]]
                try:
                    p_val = logrank_test(
                        g1[time_col], g2[time_col],
                        event_observed_A=g1[event_col],
                        event_observed_B=g2[event_col],
                    ).p_value
                except Exception:
                    p_val = None
            elif data_interval[exposure].nunique() > 2:
                try:
                    p_val = multivariate_logrank_test(
                        data_interval[time_col],
                        data_interval[exposure],
                        data_interval[event_col],
                    ).p_value
                except Exception:
                    p_val = None

    fig, ax = plt.subplots(figsize=figsize)
    kmf = KaplanMeierFitter()

    for level in data[exposure].dropna().unique():
        subset = data[data[exposure] == level]
        if subset.empty:
            continue
        kmf.fit(subset[time_col], event_observed=subset[event_col], label=str(level))
        kmf.plot(ax=ax, ci_show=True)

    ax.set_title(f"Kaplan-Meier: {outcome}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival probability", fontsize=12)
    ax.legend(title=exposure)

    if p_val is not None:
        p_text = "<0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        ax.text(0.02, 0.02, f"Log-rank p={p_text}", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 20,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create a histogram."""
    fig, ax = plt.subplots(figsize=figsize)

    if group_col and group_col != "None":
        groups = df[group_col].dropna().unique()
        colors = plt.cm.tab10.colors
        for i, g in enumerate(groups):
            sub = df[df[group_col] == g]
            ax.hist(
                sub[column].dropna(),
                bins=bins,
                alpha=0.75,
                color=colors[i % len(colors)],
                edgecolor="#111827",
                linewidth=0.6,
                label=str(g),
            )
        ax.legend(title=group_col)
    else:
        ax.hist(
            df[column].dropna(),
            bins=bins,
            alpha=0.85,
            color="#4B9EFE",
            edgecolor="#111827",
            linewidth=0.6,
        )

    ax.set_title(title or f"Histogram of {column}", fontsize=14)
    ax.set_xlabel(xlabel or column, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    return fig


def create_boxplot(
    df: pd.DataFrame,
    y_col: str,
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create a box plot."""
    fig, ax = plt.subplots(figsize=figsize)

    if group_col and group_col != "None":
        levels = df[group_col].dropna().astype(str).unique().tolist()
        values = []
        labels = []
        for level in levels:
            vals = df[df[group_col].astype(str) == level][y_col].dropna()
            if not vals.empty:
                values.append(vals.values)
                labels.append(level)
        if values:
            ax.boxplot(values, labels=labels, showfliers=True)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_title(title or f"{y_col} by {group_col}", fontsize=14)
            ax.set_xlabel(xlabel or group_col, fontsize=12)
    else:
        data = df[y_col].dropna()
        if not data.empty:
            ax.boxplot(data, labels=[y_col], showfliers=True)
            ax.set_title(title or f"{y_col} distribution", fontsize=14)
            ax.set_xlabel(xlabel or "", fontsize=12)

    ax.set_ylabel(ylabel or y_col, fontsize=12)

    plt.tight_layout()
    return fig


def create_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Create a scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col != "None":
        groups = df[hue_col].dropna().unique()
        for g in groups:
            sub = df[df[hue_col] == g]
            ax.scatter(sub[x_col], sub[y_col], label=str(g), alpha=0.7)
        ax.legend(title=hue_col)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7, color="#4B9EFE")

    ax.set_title(title or f"{x_col} vs {y_col}", fontsize=14)
    ax.set_xlabel(xlabel or x_col, fontsize=12)
    ax.set_ylabel(ylabel or y_col, fontsize=12)

    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, format: str = "png", dpi: int = 300) -> bytes:
    """Convert matplotlib figure to bytes."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
