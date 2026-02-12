"""Statistical utility functions for clinical research analysis."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _clean_variable_name(raw_name: str) -> str:
    """Convert statsmodels variable names to human-readable format.

    Examples:
        'C(hospital_volume, Treatment(reference='Low'))[T.High]' -> 'hospital_volume: High (ref: Low)'
        'C(ckd, Treatment(reference=0))[T.1]' -> 'ckd: 1 (ref: 0)'
        'C(gender)[T.Male]' -> 'gender: Male'
        'age' -> 'age'
        'Intercept' -> 'Intercept'
    """
    if raw_name == "Intercept":
        return "Intercept"

    # Pattern for C(var, Treatment(reference='ref'))[T.level] - string reference
    pattern_with_str_ref = r"C\(([^,]+),\s*Treatment\(reference=['\"]([^'\"]+)['\"]\)\)\[T\.([^\]]+)\]"
    match = re.match(pattern_with_str_ref, raw_name)
    if match:
        var_name, ref_level, level = match.groups()
        return f"{var_name}: {level} (ref: {ref_level})"

    # Pattern for C(var, Treatment(reference=0))[T.1] - numeric reference
    pattern_with_num_ref = r"C\(([^,]+),\s*Treatment\(reference=([^)]+)\)\)\[T\.([^\]]+)\]"
    match = re.match(pattern_with_num_ref, raw_name)
    if match:
        var_name, ref_level, level = match.groups()
        return f"{var_name}: {level} (ref: {ref_level})"

    # Pattern for C(var)[T.level]
    pattern_simple = r"C\(([^)]+)\)\[T\.([^\]]+)\]"
    match = re.match(pattern_simple, raw_name)
    if match:
        var_name, level = match.groups()
        return f"{var_name}: {level}"

    # Return as-is if no pattern matched
    return raw_name

try:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False


def infer_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer variable types using dtypes and cardinality heuristics."""
    inferred: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2:
                inferred[col] = "Binary"
            elif len(unique_vals) <= 10:
                inferred[col] = "Ordinal"
            else:
                inferred[col] = "Continuous"
        else:
            unique_vals = series.dropna().unique()
            if len(unique_vals) <= 2:
                inferred[col] = "Binary"
            else:
                inferred[col] = "Categorical"
    return inferred


def detect_phi_like_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns with potential identifiers/PHI by name pattern."""
    patterns = [
        "name", "dob", "dateofbirth", "birth", "mrn", "medicalrecord",
        "ssn", "address", "phone", "email", "zip", "zipcode", "id",
    ]
    matches = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "")
        if any(p in col_lower for p in patterns):
            matches.append(col)
    return matches


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness summary with percent missing per column."""
    missing = df.isna().mean().mul(100).round(2)
    return pd.DataFrame({"Percent Missing": missing}).sort_values("Percent Missing", ascending=False)


def compute_smd(df: pd.DataFrame, group_col: str, var: str, var_type: str) -> Optional[float]:
    """Compute standardized mean difference for continuous/binary/categorical variables."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return None

    g1, g2 = groups
    s1 = df[df[group_col] == g1][var]
    s2 = df[df[group_col] == g2][var]

    if var_type == "Continuous":
        s1 = pd.to_numeric(s1, errors="coerce")
        s2 = pd.to_numeric(s2, errors="coerce")
        m1, m2 = s1.mean(), s2.mean()
        sd1, sd2 = s1.std(ddof=1), s2.std(ddof=1)
        pooled = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2)
        if pooled == 0 or np.isnan(pooled):
            return None
        return float((m1 - m2) / pooled)

    if var_type in {"Binary", "Categorical", "Ordinal"}:
        levels = df[var].dropna().unique()
        if len(levels) == 0:
            return None
        smds = []
        for lvl in levels:
            p1 = (s1 == lvl).mean()
            p2 = (s2 == lvl).mean()
            pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
            if pooled == 0 or np.isnan(pooled):
                continue
            smds.append((p1 - p2) / pooled)
        if not smds:
            return None
        return float(np.max(np.abs(smds)))

    return None


def compute_vif(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute variance inflation factors for multicollinearity assessment."""
    data = df[cols].dropna().copy()
    if data.empty:
        return pd.DataFrame()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.factorize(data[col])[0]
    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.shape[1] < 2:
        return pd.DataFrame()
    vifs = []
    for i, col in enumerate(data.columns):
        vifs.append({"Variable": col, "VIF": variance_inflation_factor(data.values, i)})
    return pd.DataFrame(vifs)


def format_pvalue(p: Optional[float], threshold: float = 0.0001) -> str:
    """Format p-value for display."""
    if p is None or np.isnan(p):
        return ""
    if p < threshold:
        return "<0.0001"
    return f"{p:.4f}"


def continuous_pvalue(df: pd.DataFrame, group_col: str, var: str, nonparam: bool) -> Optional[float]:
    """Compute p-value for continuous variable comparison between groups."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return None
    g1, g2 = groups
    s1 = pd.to_numeric(df[df[group_col] == g1][var], errors="coerce").dropna()
    s2 = pd.to_numeric(df[df[group_col] == g2][var], errors="coerce").dropna()
    if nonparam:
        return stats.mannwhitneyu(s1, s2, alternative="two-sided").pvalue
    return stats.ttest_ind(s1, s2, nan_policy="omit").pvalue


def categorical_pvalue(df: pd.DataFrame, group_col: str, var: str) -> Optional[float]:
    """Compute p-value for categorical variable comparison between groups."""
    table = pd.crosstab(df[group_col], df[var])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    try:
        if table.shape == (2, 2):
            return stats.fisher_exact(table)[1]
        return stats.chi2_contingency(table)[1]
    except Exception:
        return None


def coerce_binary(series: pd.Series) -> Tuple[pd.Series, Optional[Dict[Any, int]]]:
    """Coerce a binary series to 0/1 if needed. Returns mapped series and mapping."""
    clean = series.dropna()
    unique_vals = list(pd.unique(clean))
    if len(unique_vals) != 2:
        return series, None
    if set(unique_vals) <= {0, 1}:
        return series, None
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    return series.map(mapping), mapping


def ensure_binary(series: pd.Series) -> pd.Series:
    """Validate and coerce a binary series to 0/1; raise if invalid."""
    coerced, mapping = coerce_binary(series)
    clean = coerced.dropna()
    if clean.nunique() != 2 or not set(pd.unique(clean)).issubset({0, 1}):
        raise ValueError("Binary outcome must have exactly two levels (0/1 or two categories).")
    return coerced


def fit_model(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: List[str],
    outcome_type: str,
    missing_strategy: str,
    data_dictionary: Dict[str, Any],
    exposure_reference: Optional[str] = None,
    time_event_cols: Optional[Dict[str, Dict[str, str]]] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[pd.DataFrame, str, int]:
    """Fit regression model and return result table and notes."""
    data = df.copy()
    note = ""
    dd = data_dictionary
    ref_level = exposure_reference
    if ref_level is not None and exposure in data.columns:
        levels = data[exposure].dropna().unique().tolist()
        if ref_level not in levels:
            ref_level = None

    if progress_callback:
        progress_callback(0.1, "Preparing data...")

    # Apply ordinal ordering if specified
    for col, meta in dd.items():
        if col in data.columns and meta.get("type") == "Ordinal":
            order = [v.strip() for v in meta.get("order", "").split(",") if v.strip()]
            if order:
                data[col] = pd.Categorical(data[col], categories=order, ordered=True)

    if missing_strategy == "Complete-case":
        data = data.dropna(subset=[outcome, exposure] + covariates)
        note = "Complete-case analysis; rows with missing data were removed."
    elif missing_strategy == "Simple imputation":
        for col in [outcome, exposure] + covariates:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "Missing")
        note = "Simple median/mode imputation applied; interpret with caution."

    if progress_callback:
        progress_callback(0.3, "Building model formula...")

    if outcome_type == "Binary":
        data[outcome], mapping = coerce_binary(data[outcome])
        if mapping:
            note = (note + " " if note else "") + f"Binary outcome coerced to 0/1 using mapping: {mapping}."
        clean = data[outcome].dropna()
        if clean.nunique() != 2 or not set(pd.unique(clean)).issubset({0, 1}):
            return pd.DataFrame(), "Binary outcome must have exactly two levels (0/1 or two categories).", 0
        exposure_term = exposure
        if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
            levels = data[exposure].dropna().unique().tolist()
            ref_level = ref_level if ref_level in levels else None
        if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
            exposure_term = f"C({exposure}, Treatment(reference={repr(ref_level)}))"
        cov_terms = []
        for cov in covariates:
            cov_ref = dd.get(cov, {}).get("ref_level") or None
            if dd.get(cov, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and cov_ref is not None:
                cov_levels = data[cov].dropna().unique().tolist()
                cov_ref = cov_ref if cov_ref in cov_levels else None
            if dd.get(cov, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and cov_ref is not None:
                cov_terms.append(f"C({cov}, Treatment(reference={repr(cov_ref)}))")
            else:
                cov_terms.append(cov)
        formula = f"{outcome} ~ {exposure_term}"
        if cov_terms:
            formula += " + " + " + ".join(cov_terms)

        if progress_callback:
            progress_callback(0.5, "Fitting logistic regression...")

        model = smf.logit(formula=formula, data=data).fit(disp=False)
        params = model.params
        conf = model.conf_int()
        results = pd.DataFrame({
            "Variable": [_clean_variable_name(v) for v in params.index],
            "OR": np.exp(params).values,
            "CI Lower": np.exp(conf[0]).values,
            "CI Upper": np.exp(conf[1]).values,
            "p-value": model.pvalues.apply(format_pvalue).values,
        })

        if progress_callback:
            progress_callback(1.0, "Complete")

        return results, note, len(data)

    if outcome_type == "Continuous":
        exposure_term = exposure
        if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
            levels = data[exposure].dropna().unique().tolist()
            ref_level = ref_level if ref_level in levels else None
        if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
            exposure_term = f"C({exposure}, Treatment(reference={repr(ref_level)}))"
        cov_terms = []
        for cov in covariates:
            cov_ref = dd.get(cov, {}).get("ref_level") or None
            if dd.get(cov, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and cov_ref is not None:
                cov_levels = data[cov].dropna().unique().tolist()
                cov_ref = cov_ref if cov_ref in cov_levels else None
            if dd.get(cov, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and cov_ref is not None:
                cov_terms.append(f"C({cov}, Treatment(reference={repr(cov_ref)}))")
            else:
                cov_terms.append(cov)
        formula = f"{outcome} ~ {exposure_term}"
        if cov_terms:
            formula += " + " + " + ".join(cov_terms)

        if progress_callback:
            progress_callback(0.5, "Fitting linear regression...")

        model = smf.ols(formula=formula, data=data).fit()
        params = model.params
        conf = model.conf_int()
        results = pd.DataFrame({
            "Variable": [_clean_variable_name(v) for v in params.index],
            "Beta": params.values,
            "CI Lower": conf[0].values,
            "CI Upper": conf[1].values,
            "p-value": model.pvalues.apply(format_pvalue).values,
        })

        if progress_callback:
            progress_callback(1.0, "Complete")

        return results, note, len(data)

    if outcome_type == "Time-to-event" and LIFELINES_AVAILABLE:
        if progress_callback:
            progress_callback(0.3, "Preparing survival data...")

        cph = CoxPHFitter()
        time_event = time_event_cols or {}
        mapping = time_event.get(outcome, {})
        time_col = mapping.get("time", outcome + "_time")
        event_col = mapping.get("event", outcome + "_event")
        cols = [time_col, event_col, exposure] + covariates
        missing_cols = [c for c in cols if c not in data.columns]
        if missing_cols:
            return pd.DataFrame(), f"Missing required time-to-event columns: {missing_cols}.", 0
        data = data.dropna(subset=cols)
        model_df = data[cols].copy()
        cat_types = {"Binary", "Categorical", "Ordinal"}
        cat_cols = []
        for col in [exposure] + covariates:
            if col not in model_df.columns:
                continue
            declared = dd.get(col, {}).get("type")
            if declared in cat_types or not pd.api.types.is_numeric_dtype(model_df[col]):
                cat_cols.append(col)
                if not pd.api.types.is_numeric_dtype(model_df[col]):
                    model_df[col] = model_df[col].astype(str)
        if cat_cols:
            model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
        zero_var = [c for c in model_df.columns if c not in {time_col, event_col} and model_df[c].nunique(dropna=True) <= 1]
        if zero_var:
            model_df = model_df.drop(columns=zero_var)
            note = (note + " " if note else "") + f"Dropped zero-variance columns: {zero_var}."

        if progress_callback:
            progress_callback(0.6, "Fitting Cox model...")

        try:
            cph.fit(model_df, duration_col=time_col, event_col=event_col)
        except ConvergenceError:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(model_df, duration_col=time_col, event_col=event_col)
            note = (note + " " if note else "") + "Cox model required penalization (penalizer=0.1) for convergence."
        summary = cph.summary.reset_index()
        if "covariate" in summary.columns:
            var_col = "covariate"
        elif "index" in summary.columns:
            var_col = "index"
        else:
            var_col = summary.columns[0]
        results = pd.DataFrame({
            "Variable": [_clean_variable_name(v) for v in summary[var_col]],
            "HR": summary["exp(coef)"].values,
            "CI Lower": summary["exp(coef) lower 95%"].values,
            "CI Upper": summary["exp(coef) upper 95%"].values,
            "p-value": summary["p"].apply(format_pvalue).values,
        })

        if progress_callback:
            progress_callback(1.0, "Complete")

        return results, note, len(model_df)

    return pd.DataFrame(), "Model type not supported or lifelines missing.", 0


def bootstrap_ci(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: List[str],
    outcome_type: str,
    n_boot: int,
    data_dictionary: Dict[str, Any],
    exposure_reference: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """Compute bootstrap confidence intervals."""
    dd = data_dictionary
    ref_level = exposure_reference
    exposure_term = exposure
    if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
        exposure_term = f"C({exposure}, Treatment(reference='{ref_level}'))"
    formula = f"{outcome} ~ {exposure_term}"
    if covariates:
        formula += " + " + " + ".join(covariates)

    terms: Dict[str, List[float]] = {}
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        if progress_callback and i % 20 == 0:
            progress_callback(i / n_boot, f"Bootstrap iteration {i}/{n_boot}")
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        try:
            if outcome_type == "Binary":
                model = smf.logit(formula=formula, data=sample).fit(disp=False)
                params = np.exp(model.params)
            else:
                model = smf.ols(formula=formula, data=sample).fit()
                params = model.params
            for term, val in params.items():
                if term == "Intercept":
                    continue
                terms.setdefault(term, []).append(val)
        except Exception:
            continue

    if progress_callback:
        progress_callback(1.0, "Complete")

    rows = []
    for term, values in terms.items():
        if len(values) < max(20, n_boot * 0.5):
            continue
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({"Term": _clean_variable_name(term), "Boot CI Lower": low, "Boot CI Upper": high})
    return pd.DataFrame(rows)


def assumption_warnings(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: List[str],
    outcome_type: str,
    time_event_cols: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[str]:
    """Check model assumptions and return warnings."""
    warnings = []
    if outcome_type == "Binary":
        series = df[outcome].dropna()
        coerced, _ = coerce_binary(series)
        clean = coerced.dropna()
        events = (clean == 1).sum() if set(pd.unique(clean)).issubset({0, 1}) else None
        if events is not None:
            epv = events / max(len(covariates) + 1, 1)
            if epv < 10:
                warnings.append(f"Low events-per-variable (EPV={epv:.1f}); estimates may be unstable.")
        ct = pd.crosstab(df[exposure], df[outcome])
        if (ct == 0).any().any():
            warnings.append("Zero cells detected; risk of separation.")
    if outcome_type == "Time-to-event":
        time_event = time_event_cols or {}
        time_col = time_event.get(outcome, {}).get("time", outcome + "_time")
        event_col = time_event.get(outcome, {}).get("event", outcome + "_event")
        if event_col in df.columns:
            events_series = pd.to_numeric(df[event_col], errors="coerce").dropna()
            events = events_series.sum() if not events_series.empty else None
            epv = events / max(len(covariates) + 1, 1) if events is not None else None
            if epv is not None and epv < 10:
                warnings.append(f"Low events-per-variable (EPV={epv:.1f}); Cox model may be unstable.")
    rare_levels = []
    for col in [exposure] + covariates:
        if col in df.columns and df[col].nunique(dropna=True) <= 10:
            counts = df[col].value_counts(dropna=True)
            if (counts < 5).any():
                rare_levels.append(col)
    if rare_levels:
        warnings.append(f"Rare category levels (<5) in: {rare_levels}.")
    return warnings


def extract_exposure_effects(model: Any, exposure: str, outcome: str, effect_type: str) -> List[Dict[str, Any]]:
    """Extract exposure effects from a fitted model."""
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    rows: List[Dict[str, Any]] = []

    terms = [
        t for t in params.index
        if t == exposure or t.startswith(f"C({exposure})") or t.startswith(f"C({exposure},")
    ]
    for term in terms:
        label = _clean_variable_name(term)
        est = params[term]
        ci_low = conf.loc[term, 0]
        ci_high = conf.loc[term, 1]
        p_val = pvals[term]
        if effect_type == "OR":
            rows.append({
                "Outcome": outcome,
                "Exposure Level": label,
                "Crude OR": np.exp(est),
                "CI Lower": np.exp(ci_low),
                "CI Upper": np.exp(ci_high),
                "p-value": format_pvalue(p_val),
            })
        else:
            rows.append({
                "Outcome": outcome,
                "Exposure Level": label,
                "Beta": est,
                "CI Lower": ci_low,
                "CI Upper": ci_high,
                "p-value": format_pvalue(p_val),
            })
    return rows
