"""Table generation utilities for clinical research analysis."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .statistics import (
    compute_smd,
    continuous_pvalue,
    categorical_pvalue,
    format_pvalue,
    coerce_binary,
    extract_exposure_effects,
)

try:
    from tableone import TableOne
    TABLEONE_AVAILABLE = True
except Exception:
    TABLEONE_AVAILABLE = False

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

import statsmodels.formula.api as smf


def format_cat_summary(counts: pd.Series, total: int, dec_pct: int) -> str:
    """Format categorical summary for display."""
    if total == 0:
        return ""
    top = counts.index[0] if not counts.empty else ""
    cnt = counts.iloc[0] if not counts.empty else 0
    return f"{top}: {cnt} ({cnt / total * 100:.{dec_pct}f}%)"


@st.cache_data(show_spinner=False)
def generate_table1(
    df: pd.DataFrame,
    exposure: str,
    selected_vars: List[str],
    continuous_summary: str,
    nonnormal: List[str],
    show_missing: bool,
    missing_as_category: bool,
    dec_cont: int,
    dec_pct: int,
    p_threshold: float,
    show_levels: str,
    use_tableone: bool,
    data_dictionary: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Table 1 (baseline characteristics)."""
    data = df.copy()

    if missing_as_category:
        for col in selected_vars:
            if data_dictionary[col]["type"] in {"Categorical", "Binary", "Ordinal"}:
                data[col] = data[col].fillna("Missing")

    categorical = [v for v in selected_vars if data_dictionary[v]["type"] in {"Categorical", "Binary", "Ordinal"}]
    continuous = [v for v in selected_vars if data_dictionary[v]["type"] == "Continuous"]

    raw_table = pd.DataFrame()
    if use_tableone and TABLEONE_AVAILABLE:
        table = TableOne(
            data,
            columns=selected_vars,
            categorical=categorical,
            groupby=exposure,
            nonnormal=nonnormal,
            pval=True,
        )
        raw_table = table.tableone.reset_index()

    pub_rows = []
    group_levels = data[exposure].dropna().unique().tolist()
    group_headers = ["Overall"] + [f"{lvl} (N={(data[exposure] == lvl).sum()})" for lvl in group_levels]

    for var in selected_vars:
        var_type = data_dictionary[var]["type"]
        label = data_dictionary[var]["label"]
        units = data_dictionary[var]["units"]
        display_name = label + (f" ({units})" if units else "")

        if var_type == "Continuous":
            overall = data[var].dropna()
            if var in nonnormal or continuous_summary == "Median (IQR)":
                overall_text = f"{overall.median():.{dec_cont}f} ({overall.quantile(0.25):.{dec_cont}f}, {overall.quantile(0.75):.{dec_cont}f})"
            else:
                overall_text = f"{overall.mean():.{dec_cont}f} ({overall.std(ddof=1):.{dec_cont}f})"

            row = {"Variable": display_name, "Overall": overall_text}
            p_val = continuous_pvalue(data, exposure, var, var in nonnormal or continuous_summary == "Median (IQR)")
            smd = compute_smd(data, exposure, var, var_type)

            for lvl in group_levels:
                subset = data[data[exposure] == lvl][var].dropna()
                if var in nonnormal or continuous_summary == "Median (IQR)":
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = (
                        f"{subset.median():.{dec_cont}f} ({subset.quantile(0.25):.{dec_cont}f}, {subset.quantile(0.75):.{dec_cont}f})"
                    )
                else:
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = f"{subset.mean():.{dec_cont}f} ({subset.std(ddof=1):.{dec_cont}f})"

            row["p-value"] = format_pvalue(p_val)
            row["SMD"] = round(smd, 3) if smd is not None else ""
            pub_rows.append(row)

        else:
            overall_counts = data[var].value_counts(dropna=not missing_as_category)
            p_val = categorical_pvalue(data, exposure, var)
            smd = compute_smd(data, exposure, var, var_type)

            if show_levels == "Top-level only":
                row = {"Variable": display_name}
                row["Overall"] = format_cat_summary(overall_counts, len(data), dec_pct)
                for lvl in group_levels:
                    counts = data[data[exposure] == lvl][var].value_counts(dropna=not missing_as_category)
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = format_cat_summary(counts, (data[exposure] == lvl).sum(), dec_pct)
                row["p-value"] = format_pvalue(p_val)
                row["SMD"] = round(smd, 3) if smd is not None else ""
                pub_rows.append(row)
            else:
                header = {"Variable": display_name, "Overall": "", "p-value": format_pvalue(p_val), "SMD": round(smd, 3) if smd is not None else ""}
                for lvl in group_levels:
                    header[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = ""
                pub_rows.append(header)

                for level, count in overall_counts.items():
                    row = {"Variable": f"  {level}", "Overall": f"{count} ({count / len(data) * 100:.{dec_pct}f}%)"}
                    for lvl in group_levels:
                        subset = data[data[exposure] == lvl]
                        cnt = (subset[var] == level).sum()
                        row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = (
                            f"{cnt} ({cnt / max((data[exposure] == lvl).sum(), 1) * 100:.{dec_pct}f}%)"
                        )
                    row["p-value"] = ""
                    row["SMD"] = ""
                    pub_rows.append(row)

    pub_table = pd.DataFrame(pub_rows)
    pub_table = pub_table[["Variable"] + group_headers + ["p-value", "SMD"]]
    return pub_table, raw_table


@st.cache_data(show_spinner=False)
def generate_table2(
    df: pd.DataFrame,
    exposure: str,
    outcomes: List[str],
    outcome_types: Dict[str, str],
    use_tableone: bool,
    time_event_cols: Dict[str, Dict[str, str]],
    cont_summary: str,
    cont_dec: int,
    pct_dec: int,
    data_dictionary: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Generate Table 2 (outcome analysis)."""
    data = df.copy()
    rows = []
    raw_table = None

    if use_tableone and TABLEONE_AVAILABLE:
        categorical = [o for o in outcomes if outcome_types[o] in {"Binary", "Categorical", "Ordinal"}]
        nonnormal = [o for o in outcomes if outcome_types[o] == "Continuous"]
        table = TableOne(
            data,
            columns=outcomes,
            categorical=categorical,
            groupby=exposure,
            nonnormal=nonnormal,
            pval=True,
        )
        raw_table = table.tableone.reset_index()

    group_levels = data[exposure].dropna().unique().tolist()
    group_headers = ["Overall"] + [f"{lvl} (N={(data[exposure] == lvl).sum()})" for lvl in group_levels]

    for outcome in outcomes:
        var_type = outcome_types[outcome]
        label = data_dictionary[outcome]["label"]
        units = data_dictionary[outcome]["units"]
        display_name = label + (f" ({units})" if units else "")

        if var_type == "Continuous":
            overall = pd.to_numeric(data[outcome], errors="coerce").dropna()
            if cont_summary == "Median (IQR)":
                overall_text = (
                    f"{overall.median():.{cont_dec}f} "
                    f"({overall.quantile(0.25):.{cont_dec}f}, {overall.quantile(0.75):.{cont_dec}f})"
                )
            else:
                overall_text = f"{overall.mean():.{cont_dec}f} ({overall.std(ddof=1):.{cont_dec}f})"
            row = {"Variable": display_name, "Overall": overall_text}
            p_val = continuous_pvalue(data, exposure, outcome, nonparam=(cont_summary == "Median (IQR)"))
            for lvl in group_levels:
                subset = pd.to_numeric(data[data[exposure] == lvl][outcome], errors="coerce").dropna()
                if cont_summary == "Median (IQR)":
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = (
                        f"{subset.median():.{cont_dec}f} "
                        f"({subset.quantile(0.25):.{cont_dec}f}, {subset.quantile(0.75):.{cont_dec}f})"
                    )
                else:
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = (
                        f"{subset.mean():.{cont_dec}f} ({subset.std(ddof=1):.{cont_dec}f})"
                    )
            row["p-value"] = format_pvalue(p_val)
            rows.append(row)
        elif var_type in {"Binary", "Categorical", "Ordinal"}:
            overall_counts = data[outcome].value_counts(dropna=True)
            p_val = categorical_pvalue(data, exposure, outcome)
            header = {"Variable": display_name, "Overall": "", "p-value": format_pvalue(p_val)}
            for lvl in group_levels:
                header[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = ""
            rows.append(header)
            for level, count in overall_counts.items():
                row = {"Variable": f"  {level}", "Overall": f"{count} ({count / len(data) * 100:.{pct_dec}f}%)"}
                for lvl in group_levels:
                    subset = data[data[exposure] == lvl]
                    cnt = (subset[outcome] == level).sum()
                    row[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = (
                        f"{cnt} ({cnt / max((data[exposure] == lvl).sum(), 1) * 100:.{pct_dec}f}%)"
                    )
                row["p-value"] = ""
                rows.append(row)
        elif var_type == "Time-to-event" and LIFELINES_AVAILABLE:
            mapping = time_event_cols.get(outcome, {})
            time_col = mapping.get("time", f"{outcome}_time")
            event_col = mapping.get("event", f"{outcome}_event")
            if time_col not in data.columns or event_col not in data.columns:
                rows.append({"Variable": display_name, "Overall": "", "p-value": "", "Note": "Missing time/event columns."})
                continue
            cph = CoxPHFitter()
            temp = data[[time_col, event_col, exposure]].dropna()
            if not pd.api.types.is_numeric_dtype(temp[exposure]):
                temp = pd.get_dummies(temp, columns=[exposure], drop_first=True)
            if temp.empty:
                rows.append({"Variable": display_name, "Overall": "", "p-value": "", "Note": "No complete rows for time/event analysis."})
                continue
            cph.fit(temp, duration_col=time_col, event_col=event_col)
            summary = cph.summary.reset_index()
            if "term" in summary.columns:
                term_col = "term"
            elif "covariate" in summary.columns:
                term_col = "covariate"
            elif "index" in summary.columns:
                term_col = "index"
            else:
                term_col = summary.columns[0]
            header = {"Variable": display_name, "Overall": "", "p-value": ""}
            for lvl in group_levels:
                header[f"{lvl} (N={(data[exposure] == lvl).sum()})"] = ""
            rows.append(header)
            for _, row in summary.iterrows():
                rows.append({
                    "Variable": f"  {row[term_col]}",
                    "Overall": f"HR {row['exp(coef)']:.2f} ({row['exp(coef) lower 95%']:.2f}, {row['exp(coef) upper 95%']:.2f})",
                    "p-value": format_pvalue(row["p"]),
                })

    table2 = pd.DataFrame(rows)
    if not table2.empty:
        table2 = table2[["Variable"] + group_headers + (["p-value"] if "p-value" in table2.columns else [])]
    return table2, raw_table


def generate_or_table(
    df: pd.DataFrame,
    exposure: str,
    outcomes: List[str],
    outcome_types: Dict[str, str],
    data_dictionary: Dict[str, Any],
    exposure_reference: Optional[str] = None,
) -> pd.DataFrame:
    """Generate crude odds ratio table for binary outcomes."""
    rows: List[Dict[str, Any]] = []
    dd = data_dictionary
    ref_level = exposure_reference
    for outcome in outcomes:
        if outcome_types[outcome] != "Binary":
            continue
        data = df.copy()
        data[outcome], mapping = coerce_binary(data[outcome])
        note = f"Outcome mapped to 0/1: {mapping}." if mapping else ""
        clean = data[outcome].dropna()
        if clean.nunique() != 2 or not set(pd.unique(clean)).issubset({0, 1}):
            rows.append({"Outcome": outcome, "Note": "Binary outcome must have exactly two levels."})
            continue
        if ref_level is not None and exposure in data.columns:
            levels = data[exposure].dropna().unique().tolist()
            if ref_level not in levels:
                ref_level = None
        exposure_term = f"C({exposure})"
        if dd.get(exposure, {}).get("type") in {"Binary", "Categorical", "Ordinal"} and ref_level is not None:
            exposure_term = f"C({exposure}, Treatment(reference={repr(ref_level)}))"
        model = smf.logit(f"{outcome} ~ {exposure_term}", data=data).fit(disp=False)
        for row in extract_exposure_effects(model, exposure, outcome, effect_type="OR"):
            if note:
                row["Note"] = note
            rows.append(row)
    return pd.DataFrame(rows)
