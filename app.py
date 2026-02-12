"""Clinical Research Analysis (Beta) - Streamlit Application.

A comprehensive tool for epidemiological and biostatistical analyses on clinical datasets.
"""

import json
import random
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import utilities from modular structure
from utils.statistics import (
    infer_variable_types,
    detect_phi_like_columns,
    summarize_missingness,
    compute_smd,
    compute_vif,
    format_pvalue,
    fit_model,
    bootstrap_ci,
    assumption_warnings,
    coerce_binary,
    LIFELINES_AVAILABLE,
)
from utils.tables import (
    generate_table1,
    generate_table2,
    generate_or_table,
    TABLEONE_AVAILABLE,
)
from utils.export import (
    to_excel_bytes,
    round_numeric_columns,
    clean_regression_results,
    build_run_config_json,
    generate_word_report,
    DOCX_AVAILABLE,
)
from utils.transforms import (
    apply_log_transform,
    apply_sqrt_transform,
    apply_zscore_transform,
    categorize_continuous,
    dichotomize,
    combine_categories,
    detect_outliers,
    winsorize,
    create_interaction,
    get_transformation_summary,
)
from utils.plots import (
    create_forest_plot,
    create_km_plot,
    create_histogram,
    create_boxplot,
    create_scatter,
    fig_to_bytes,
)
from utils.history import (
    get_history,
    save_analysis_state,
    undo_action,
    redo_action,
    log_event,
)

try:
    from lifelines import KaplanMeierFitter
except Exception:
    pass

APP_VERSION = "0.2.0"

ROLE_OPTIONS = [
    "Exposure",
    "Outcome",
    "Baseline covariate",
    "Post-exposure",
    "Identifier",
    "Ignore",
]

TYPE_OPTIONS = [
    "Continuous",
    "Binary",
    "Categorical",
    "Ordinal",
    "Time-to-event component",
]


# -----------------------------
# Session State Initialization
# -----------------------------

def _init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "analysis_log": [],
        "time_event_cols": {},
        "progress": {},
        "df": None,
        "raw_df": None,
        "merge_history": {},
        "filter_conditions": [],
        "transformation_history": [],
        "analysis_ready": False,  # Flag to indicate user is ready for analysis
        "new_columns": [],  # Track columns created through transformations
        "original_columns": [],  # Track original columns from uploaded data
        "data_loaded_from": None,  # Track which file/source data was loaded from
        "pending_toast": None,  # Toast message to show after rerun
        "theme": "light",  # Theme: "light" or "dark"
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _get_theme_css() -> str:
    """Get CSS styles based on current theme."""
    is_dark = st.session_state.get("theme", "light") == "dark"

    if is_dark:
        # Dark theme - Comprehensive professional dark theme
        return """
        <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-input: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --accent-color: #3b82f6;
            --accent-hover: #60a5fa;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
        }

        /* Global dark background */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
        .main, .block-container, [data-testid="stVerticalBlock"],
        section.main, .stApp {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }

        /* Main content area */
        [data-testid="stAppViewContainer"] > section > div {
            background-color: var(--bg-primary) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        [data-testid="stSidebar"] *, section[data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        /* All text elements */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
        .stText, p, span, label, .stCaption, div, li, td, th {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5, h6, .stSubheader, .stHeader, .stTitle {
            color: var(--text-primary) !important;
        }

        /* Buttons */
        div.stButton > button, .stDownloadButton > button {
            background-color: var(--accent-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }

        div.stButton > button:hover, .stDownloadButton > button:hover {
            background-color: var(--accent-hover) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
        }

        /* All form inputs */
        input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
        }

        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Selectbox and multiselect */
        .stSelectbox, .stMultiSelect {
            color: var(--text-primary) !important;
        }

        .stSelectbox > div > div, .stMultiSelect > div > div,
        [data-baseweb="select"] > div, [data-baseweb="popover"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Dropdown menus */
        [data-baseweb="menu"], [data-baseweb="popover"] > div,
        [role="listbox"], ul[role="listbox"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
        }

        [data-baseweb="menu"] li, [role="option"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        [data-baseweb="menu"] li:hover, [role="option"]:hover {
            background-color: var(--bg-tertiary) !important;
        }

        /* Selected items in multiselect */
        [data-baseweb="tag"] {
            background-color: var(--accent-color) !important;
            color: white !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--bg-secondary) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            border: 1px solid var(--border-color) !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color) !important;
            color: white !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 20px 10px !important;
            background-color: var(--bg-primary) !important;
        }

        /* Expanders */
        .streamlit-expanderHeader, details summary {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        .streamlit-expanderContent, details > div {
            background-color: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
        }

        [data-testid="stExpander"] {
            background-color: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        /* DataFrames and tables */
        [data-testid="stDataFrame"], .stDataFrame,
        [data-testid="stTable"], .stTable {
            background-color: var(--bg-secondary) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border-color) !important;
        }

        [data-testid="stDataFrame"] *, .stDataFrame *,
        table, table *, thead, tbody, tr, td, th {
            color: var(--text-primary) !important;
            background-color: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }

        /* Slider */
        .stSlider > div > div {
            background-color: var(--bg-tertiary) !important;
        }

        .stSlider [data-baseweb="slider"] {
            background-color: var(--bg-tertiary) !important;
        }

        /* Metrics */
        [data-testid="stMetric"], [data-testid="metric-container"] {
            background-color: var(--bg-secondary) !important;
            padding: 16px !important;
            border-radius: 10px !important;
            border: 1px solid var(--border-color) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }

        /* Alerts and info boxes */
        .stAlert, [data-testid="stAlert"],
        .element-container .stAlert {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        .stAlert p, [data-testid="stAlert"] p {
            color: var(--text-primary) !important;
        }

        /* Info, warning, error, success boxes */
        [data-testid="stNotification"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        /* Checkbox and Radio */
        .stCheckbox, .stRadio, .stCheckbox label, .stRadio label {
            color: var(--text-primary) !important;
        }

        .stCheckbox > label > span, .stRadio > label > span {
            color: var(--text-primary) !important;
        }

        /* Toggle */
        [data-testid="stToggle"] span {
            color: var(--text-primary) !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"], .stFileUploader {
            background-color: var(--bg-secondary) !important;
            border: 1px dashed var(--border-color) !important;
            border-radius: 8px !important;
        }

        [data-testid="stFileUploader"] * {
            color: var(--text-primary) !important;
        }

        /* Code blocks */
        .stCodeBlock, code, pre {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Divider */
        hr, [data-testid="stSeparator"] {
            border-color: var(--border-color) !important;
            background-color: var(--border-color) !important;
        }

        /* Progress bar */
        .stProgress > div > div {
            background-color: var(--bg-tertiary) !important;
        }

        .stProgress > div > div > div {
            background-color: var(--accent-color) !important;
        }

        /* Tooltips */
        [data-baseweb="tooltip"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }

        /* JSON display */
        .stJson {
            background-color: var(--bg-secondary) !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Plots/Charts background */
        .stPlotlyChart, [data-testid="stPlotlyChart"],
        .js-plotly-plot, .plot-container {
            background-color: var(--bg-secondary) !important;
            border-radius: 8px !important;
        }

        /* ===== STREAMLIT TOOLBAR/HEADER BAR ===== */
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        .stApp > header,
        header {
            background-color: var(--bg-secondary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }

        [data-testid="stHeader"] *, header * {
            color: var(--text-primary) !important;
        }

        /* Toolbar buttons */
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] button,
        .stToolbar, .stToolbar button {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        /* Status widget (running indicator) */
        [data-testid="stStatusWidget"],
        .stStatusWidget {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        /* Decoration (top bar) */
        [data-testid="stDecoration"],
        .stDecoration {
            background-color: var(--bg-secondary) !important;
        }

        /* ===== FILE UPLOADER - COMPREHENSIVE FIX ===== */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] > section,
        .stFileUploader,
        .uploadedFile {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] button span,
        .stFileUploader * {
            color: var(--text-primary) !important;
            background-color: transparent !important;
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploadDropzone"] button {
            background-color: var(--accent-color) !important;
            color: white !important;
            border: none !important;
        }

        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileDropzoneInstructions"] {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stFileDropzoneInstructions"] span,
        [data-testid="stFileDropzoneInstructions"] div {
            color: var(--text-secondary) !important;
        }

        /* ===== DATAFRAME/TABLE - COMPREHENSIVE FIX ===== */
        /* Apply invert filter to canvas-based dataframes for dark mode */
        [data-testid="stDataFrame"] canvas {
            filter: invert(1) hue-rotate(180deg) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div,
        [data-testid="stTable"],
        .stDataFrame {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        /* Wrapper for dataframe */
        [data-testid="stDataFrame"] > div:first-child {
            background-color: var(--bg-secondary) !important;
        }

        /* DataFrame resize handle */
        [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
            background-color: var(--bg-secondary) !important;
        }

        /* Table element styling (for st.table) */
        table {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-collapse: collapse !important;
            width: 100% !important;
        }

        table th {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            padding: 10px 14px !important;
            font-weight: 600 !important;
            text-align: left !important;
        }

        table td {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            padding: 8px 14px !important;
        }

        table tr:nth-child(even) td {
            background-color: rgba(51, 65, 85, 0.5) !important;
        }

        table tr:hover td {
            background-color: var(--bg-tertiary) !important;
        }

        /* Dataframe toolbar */
        [data-testid="stElementToolbar"] {
            background-color: var(--bg-tertiary) !important;
        }

        [data-testid="stElementToolbar"] button {
            color: var(--text-primary) !important;
        }

        /* ===== BOTTOM CONTAINER/FOOTER ===== */
        [data-testid="stBottom"],
        .stBottom {
            background-color: var(--bg-primary) !important;
        }

        /* ===== MAIN BLOCK CONTAINERS ===== */
        [data-testid="stVerticalBlockBorderWrapper"],
        .stVerticalBlockBorderWrapper {
            background-color: var(--bg-primary) !important;
        }

        [data-testid="stHorizontalBlock"],
        .stHorizontalBlock {
            background-color: transparent !important;
        }
        </style>
        """
    else:
        # Light theme - Clean, professional light
        return """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --accent-color: #4B9EFE;
            --accent-hover: #3A8AE8;
            --success-color: #059669;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text-primary) !important;
        }

        .stMarkdown, .stText, p, span, label, .stCaption {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5, h6, .stSubheader, .stHeader, .stTitle {
            color: var(--text-primary) !important;
        }

        /* Buttons */
        div.stButton > button, .stDownloadButton > button {
            background-color: var(--accent-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button:hover, .stDownloadButton > button:hover {
            background-color: var(--accent-hover) !important;
            box-shadow: 0 4px 12px rgba(75, 158, 254, 0.3) !important;
        }

        div.stButton > button[kind="secondary"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--bg-secondary) !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 500 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color) !important;
            color: white !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 20px 10px !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }

        .streamlit-expanderContent {
            background-color: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* DataFrames */
        [data-testid="stDataFrame"], .stDataFrame {
            background-color: var(--bg-primary) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border-color) !important;
        }

        [data-testid="stDataFrame"] *, .stDataFrame * {
            color: var(--text-primary) !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background-color: var(--bg-secondary) !important;
            padding: 16px !important;
            border-radius: 10px !important;
            border: 1px solid var(--border-color) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }

        /* Alerts */
        .stAlert {
            border-radius: 8px !important;
        }

        /* Checkbox and Radio */
        .stCheckbox, .stRadio {
            color: var(--text-primary) !important;
        }

        /* Divider */
        hr {
            border-color: var(--border-color) !important;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        </style>
        """


def _themed_dataframe(df: pd.DataFrame, max_rows: int = 50, max_height: str = "400px") -> None:
    """Display a DataFrame as a theme-aware HTML table.

    This function renders DataFrames as styled HTML tables that properly support
    dark mode, instead of using Streamlit's canvas-based renderer which doesn't
    respond to CSS styling.

    Args:
        df: The DataFrame to display
        max_rows: Maximum number of rows to display (default 50)
        max_height: Maximum height of the scrollable container
    """
    is_dark = st.session_state.get("theme", "light") == "dark"

    # Theme colors
    if is_dark:
        bg_header = "#334155"
        bg_row_even = "#1e293b"
        bg_row_odd = "#263245"
        bg_hover = "#3b4d66"
        text_color = "#f1f5f9"
        border_color = "#475569"
        container_bg = "#1e293b"
    else:
        bg_header = "#e2e8f0"
        bg_row_even = "#ffffff"
        bg_row_odd = "#f8fafc"
        bg_hover = "#e2e8f0"
        text_color = "#0f172a"
        border_color = "#e2e8f0"
        container_bg = "#ffffff"

    # Limit rows for display
    display_df = df.head(max_rows) if len(df) > max_rows else df
    truncated = len(df) > max_rows

    # Generate unique ID for this table
    table_id = f"themed_table_{random.randint(10000, 99999)}"

    # Build HTML table
    html_parts = [f"""
    <style>
    #{table_id}-container {{
        max-height: {max_height};
        overflow: auto;
        border-radius: 8px;
        border: 1px solid {border_color};
        background-color: {container_bg};
        margin: 10px 0;
    }}
    #{table_id} {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    #{table_id} th {{
        background-color: {bg_header};
        color: {text_color};
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid {border_color};
        position: sticky;
        top: 0;
        z-index: 1;
    }}
    #{table_id} td {{
        color: {text_color};
        padding: 10px 16px;
        border-bottom: 1px solid {border_color};
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    #{table_id} tr:nth-child(even) td {{
        background-color: {bg_row_even};
    }}
    #{table_id} tr:nth-child(odd) td {{
        background-color: {bg_row_odd};
    }}
    #{table_id} tr:hover td {{
        background-color: {bg_hover};
    }}
    </style>
    <div id="{table_id}-container">
    <table id="{table_id}">
    <thead><tr>
    """]

    # Add index column header if index is meaningful
    if not isinstance(display_df.index, pd.RangeIndex) or display_df.index.name:
        index_name = display_df.index.name if display_df.index.name else ""
        html_parts.append(f"<th>{index_name}</th>")

    # Add column headers
    for col in display_df.columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead><tbody>")

    # Add data rows
    for idx, row in display_df.iterrows():
        html_parts.append("<tr>")
        # Add index value if index is meaningful
        if not isinstance(display_df.index, pd.RangeIndex) or display_df.index.name:
            html_parts.append(f"<td>{idx}</td>")
        for val in row:
            # Format the value
            if pd.isna(val):
                display_val = "<em style='color: #94a3b8;'>NaN</em>"
            elif isinstance(val, float):
                display_val = f"{val:.4g}" if abs(val) < 10000 else f"{val:.2e}"
            else:
                display_val = str(val)[:100]  # Truncate long strings
            html_parts.append(f"<td>{display_val}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")

    # Add truncation message if needed
    if truncated:
        html_parts.append(f"<p style='color: {text_color}; font-size: 12px; margin-top: 5px;'><em>Showing {max_rows} of {len(df)} rows</em></p>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _show_pending_toast() -> None:
    """Show any pending toast message - now handled inline in transformation section."""
    pass  # Toast is now shown inline in the transformation section


def _show_inline_success() -> bool:
    """Show inline success message if there's a pending toast. Returns True if shown."""
    if st.session_state.get("pending_toast"):
        message = st.session_state.pending_toast
        st.markdown(
            f"""
            <style>
            @keyframes successFadeOut {{
                0% {{ opacity: 1; }}
                70% {{ opacity: 1; }}
                100% {{ opacity: 0; }}
            }}
            .inline-success {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin: 10px 0;
                animation: successFadeOut 2.5s ease-in-out forwards;
            }}
            .inline-success svg {{
                width: 18px;
                height: 18px;
            }}
            </style>
            <div class="inline-success">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.pending_toast = None
        return True
    return False


def _queue_toast(message: str) -> None:
    """Queue a toast message to show after rerun."""
    st.session_state.pending_toast = message


def _render_sidebar_progress() -> None:
    """Render the sidebar progress panel with theme-aware colors."""
    # Theme-aware colors
    is_dark = st.session_state.get("theme", "light") == "dark"
    bg_card = "#1e293b" if is_dark else "#f8fafc"
    bg_item = "#334155" if is_dark else "#ffffff"
    border_color = "#475569" if is_dark else "#e5e7eb"
    text_primary = "#f1f5f9" if is_dark else "#111827"
    text_secondary = "#94a3b8" if is_dark else "#6b7280"

    st.sidebar.markdown(
        f"""
<div style="padding:10px 12px;border-radius:10px;background:{bg_card};border:1px solid {border_color};margin-bottom:8px;">
  <div style="font-weight:700;color:{text_primary};margin-bottom:6px;">Progress</div>
  <div style="font-size:12px;color:{text_secondary};">Track your analysis steps</div>
</div>
""",
        unsafe_allow_html=True,
    )

    total = st.session_state.get("df").shape[0] if st.session_state.get("df") is not None else 0
    t1 = st.session_state.get("table1_n", "—")
    t2 = st.session_state.get("table2_n", "—")
    reg = st.session_state.get("model_n", "—")

    st.sidebar.markdown(
        f"""
<div style="padding:8px 10px;border-radius:8px;background:{bg_item};border:1px solid {border_color};margin-bottom:8px;">
  <div style="font-size:12px;color:{text_secondary};">Status</div>
  <div style="font-size:13px;color:{text_primary};">Rows: {total}</div>
  <div style="font-size:13px;color:{text_primary};">Table 1: {t1}</div>
  <div style="font-size:13px;color:{text_primary};">Table 2: {t2}</div>
  <div style="font-size:13px;color:{text_primary};">Regression: {reg}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Data Preparation steps
    st.sidebar.markdown(
        f'<div style="font-size:11px;font-weight:600;color:{text_secondary};margin:8px 0 4px 0;text-transform:uppercase;">Data Preparation</div>',
        unsafe_allow_html=True,
    )
    prep_steps = [
        ("upload", "Upload & Inspect"),
        ("ready", "Ready for Analysis"),
    ]
    for key, label in prep_steps:
        done = st.session_state.progress.get(key, False) if key != "ready" else st.session_state.get("analysis_ready", False)
        badge_color = "#10b981" if done else "#64748b"
        badge_text = "Done" if done else "Pending"
        st.sidebar.markdown(
            f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;margin:4px 0;border-radius:8px;background:{bg_item};border:1px solid {border_color};">
  <div style="font-size:13px;color:{text_primary};">{label}</div>
  <div style="font-size:11px;padding:2px 8px;border-radius:999px;background:{badge_color};color:#ffffff;">{badge_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Analysis steps
    st.sidebar.markdown(
        f'<div style="font-size:11px;font-weight:600;color:{text_secondary};margin:12px 0 4px 0;text-transform:uppercase;">Analysis</div>',
        unsafe_allow_html=True,
    )
    analysis_steps = [
        ("dictionary", "📋 Data Dictionary"),
        ("table1", "📊 Tables"),
        ("regression", "🔬 Regression"),
    ]
    for key, label in analysis_steps:
        done = st.session_state.progress.get(key, False)
        badge_color = "#10b981" if done else "#64748b"
        badge_text = "Done" if done else "Pending"
        st.sidebar.markdown(
            f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 10px;margin:4px 0;border-radius:8px;background:{bg_item};border:1px solid {border_color};">
  <div style="font-size:13px;color:{text_primary};">{label}</div>
  <div style="font-size:11px;padding:2px 8px;border-radius:999px;background:{badge_color};color:#ffffff;">{badge_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    exposure = st.session_state.get("table1_exposure") or "Not set"
    outcomes = _get_role_vars("Outcome")
    covariates = st.session_state.get("selected_covariates", [])
    st.sidebar.markdown(
        f"""
<div style="margin-top:10px;padding:10px 12px;border-radius:10px;background:{bg_item};border:1px solid {border_color};">
  <div style="font-size:13px;font-weight:600;color:{text_primary};margin-bottom:6px;">Study Setup</div>
  <div style="font-size:12px;color:{text_primary};margin-bottom:6px;">Exposure: <b>{exposure}</b></div>
  <div style="font-size:12px;color:{text_primary};margin-bottom:6px;">Outcomes: {', '.join(outcomes) if outcomes else 'Not set'}</div>
  <div style="font-size:12px;color:{text_primary};">Covariates: {', '.join(covariates) if covariates else 'Not set'}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Undo/Redo buttons
    st.sidebar.markdown("---")
    history = get_history()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("↩ Undo", disabled=not history.can_undo(), use_container_width=True):
            if undo_action():
                st.rerun()
    with col2:
        if st.button("↪ Redo", disabled=not history.can_redo(), use_container_width=True):
            if redo_action():
                st.rerun()

    # History view
    if st.sidebar.checkbox("Show history", value=False):
        summary = history.get_history_summary()
        if summary:
            for item in reversed(summary[-10:]):
                marker = "→ " if item["current"] else "  "
                st.sidebar.text(f"{marker}{item['action'][:30]}")


def _sample_data() -> pd.DataFrame:
    """Generate sample data for demonstration."""
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(60, 12, n).round(1),
        "sex": rng.choice(["Female", "Male"], size=n, p=[0.55, 0.45]),
        "exposure": rng.choice(["Control", "Treatment"], size=n),
        "bmi": rng.normal(27, 5, n).round(1),
        "smoker": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "outcome_binary": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "outcome_cont": rng.normal(10, 3, n).round(2),
        "survival_days_1y_time": rng.integers(1, 365, size=n),
        "survival_days_1y_event": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
    })
    return df


def _get_role_vars(role: str, df: Optional[pd.DataFrame] = None) -> List[str]:
    """Get variables with a specific role from data dictionary.

    Args:
        role: The role to filter by
        df: Optional DataFrame to validate columns exist. If provided,
            only returns variables that exist in both the data dictionary and DataFrame.
    """
    dd = st.session_state.get("data_dictionary", {})
    if df is None:
        df = st.session_state.get("df")
    if df is None:
        # No DataFrame available, return all matching roles from data dictionary
        return [c for c, v in dd.items() if v["role"] == role]
    df_cols = set(df.columns)
    return [c for c, v in dd.items() if v["role"] == role and c in df_cols]


def _update_data_dictionary_for_new_columns(df: pd.DataFrame) -> None:
    """Update data dictionary to include any new columns and remove stale ones."""
    if "data_dictionary" not in st.session_state:
        return

    dd = st.session_state.data_dictionary
    inferred = infer_variable_types(df)

    # Add new columns
    for col in df.columns:
        if col not in dd:
            dd[col] = {
                "role": "Baseline covariate",
                "type": inferred.get(col, "Continuous"),
                "label": col,
                "units": "",
                "order": "",
                "ref_level": "",
            }

    # Remove columns that no longer exist in the DataFrame
    cols_to_remove = [col for col in dd.keys() if col not in df.columns]
    for col in cols_to_remove:
        del dd[col]

    st.session_state.data_dictionary = dd


def _sync_data_dictionary(df: pd.DataFrame) -> None:
    """Ensure data dictionary is in sync with DataFrame columns."""
    _update_data_dictionary_for_new_columns(df)


def _track_new_column(new_col: str) -> None:
    """Track a newly created column from transformations."""
    if "new_columns" not in st.session_state:
        st.session_state.new_columns = []
    if new_col not in st.session_state.new_columns:
        st.session_state.new_columns.append(new_col)
    # Reset analysis_ready since data has changed
    st.session_state.analysis_ready = False


def _get_new_columns() -> List[str]:
    """Get list of columns created through transformations."""
    current_df = st.session_state.get("df")
    if current_df is None:
        return []
    original = set(st.session_state.get("original_columns", []))
    current = set(current_df.columns)
    return [c for c in current if c not in original]


# -----------------------------
# UI Sections
# -----------------------------

def load_data_section() -> Optional[pd.DataFrame]:
    """Data upload and inspection section."""
    st.header("Upload & Inspect")
    st.write("Upload a cleaned, de-identified dataset for analysis.")
    show_dtypes = st.checkbox("Show column data types", value=False)

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.caption("Quick start")
        if st.button("Use sample data"):
            st.session_state.df = _sample_data()
            st.session_state.raw_df = st.session_state.df.copy()
            st.session_state.original_columns = list(st.session_state.df.columns)
            st.session_state.new_columns = []
            st.session_state.analysis_ready = False
            st.session_state.data_loaded_from = "sample"
            save_analysis_state("Loaded sample data", {"rows": len(st.session_state.df)})
            st.session_state.progress["upload"] = True
            st.rerun()
        st.caption("Try the app without uploading a file.")
    with col2:
        st.caption("Upload file")
        uploaded = st.file_uploader("Browse files", type=["csv", "xlsx"])

    # Only process upload if it's a NEW file (not already loaded)
    if uploaded is not None:
        # Check if this is a new file upload by comparing file name
        current_file = st.session_state.get("data_loaded_from")
        if current_file != uploaded.name:
            try:
                if uploaded.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded)
                else:
                    st.session_state.df = pd.read_excel(uploaded)
                st.session_state.raw_df = st.session_state.df.copy()
                st.session_state.original_columns = list(st.session_state.df.columns)
                st.session_state.new_columns = []
                st.session_state.analysis_ready = False
                st.session_state.data_loaded_from = uploaded.name
                save_analysis_state("Data uploaded", {"name": uploaded.name})
                st.session_state.progress["upload"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to read file: {exc}")
                return None

    df = st.session_state.get("df")
    if df is None:
        return None

    st.success("Data loaded successfully.")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    _themed_dataframe(df.head(), max_rows=10, max_height="300px")
    st.session_state.progress["upload"] = True

    if show_dtypes:
        st.subheader("Data types")
        _themed_dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}), max_height="250px")

    st.subheader("Percent of Data Missing:")
    miss_table = summarize_missingness(df).copy()
    miss_table["Percent Missing"] = miss_table["Percent Missing"].map(lambda x: f"{x:.2f}%")
    _themed_dataframe(miss_table, max_height="300px")

    show_categorical = st.checkbox("Show categorical previews", value=False)
    if show_categorical:
        st.subheader("Categorical previews")
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 10:
                st.write(f"{col}: {df[col].dropna().unique().tolist()}")

    st.subheader("Data readiness checklist")
    dup_rows = df.duplicated().sum()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_missing = summarize_missingness(df)
    high_missing = high_missing[high_missing["Percent Missing"] > 10].index.tolist()
    time_cols = [c for c in df.columns if c.endswith("_time")]
    negative_time = [c for c in time_cols if pd.to_numeric(df[c], errors="coerce").dropna().lt(0).any()]
    st.write(f"Duplicate rows: {dup_rows}")
    st.write(f"Constant columns: {constant_cols if constant_cols else 'None'}")
    st.write(f"High missingness (>10%): {high_missing if high_missing else 'None'}")
    st.write(f"Negative time values: {negative_time if negative_time else 'None'}")

    st.session_state.progress["upload"] = True
    return df


def data_filtering_section(df: pd.DataFrame) -> pd.DataFrame:
    """Data filtering/subsetting section."""
    st.header("Data Filtering")
    st.write("Filter data to specific subpopulations before analysis.")

    # Show current row count
    original_count = len(st.session_state.get("raw_df", df))
    current_count = len(df)
    st.info(f"Current: {current_count} rows (Original: {original_count} rows)")

    # Reset filters button
    if st.button("Reset to original data"):
        st.session_state.df = st.session_state.raw_df.copy()
        st.session_state.filter_conditions = []
        st.session_state.new_columns = []
        st.session_state.original_columns = list(st.session_state.raw_df.columns)
        st.session_state.analysis_ready = False
        _sync_data_dictionary(st.session_state.df)
        save_analysis_state("Reset filters")
        st.rerun()

    # Add new filter
    st.subheader("Add Filter")
    cols = df.columns.tolist()
    filter_col = st.selectbox("Column to filter", cols, key="filter_col")

    if filter_col:
        col_data = df[filter_col]

        if pd.api.types.is_numeric_dtype(col_data):
            filter_type = st.radio("Filter type", ["Range", "Comparison", "Exclude missing"], horizontal=True)

            if filter_type == "Range":
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                range_vals = st.slider(
                    f"Select range for {filter_col}",
                    min_val, max_val, (min_val, max_val)
                )
                if st.button("Apply range filter"):
                    mask = (df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])
                    st.session_state.df = df[mask].copy()
                    st.session_state.filter_conditions.append(
                        f"{filter_col} between {range_vals[0]} and {range_vals[1]}"
                    )
                    st.session_state.analysis_ready = False
                    save_analysis_state("Applied range filter", {"column": filter_col, "range": range_vals})
                    st.rerun()

            elif filter_type == "Comparison":
                operator_labels = {
                    "Equal to (=)": "==",
                    "Not equal to (≠)": "!=",
                    "Greater than (>)": ">",
                    "Greater than or equal (≥)": ">=",
                    "Less than (<)": "<",
                    "Less than or equal (≤)": "<=",
                }
                comp_label = st.selectbox("Operator", list(operator_labels.keys()))
                comp_op = operator_labels[comp_label]
                comp_val = st.number_input("Value", value=float(col_data.median()))
                if st.button("Apply comparison filter"):
                    if comp_op == "==":
                        mask = df[filter_col] == comp_val
                    elif comp_op == "!=":
                        mask = df[filter_col] != comp_val
                    elif comp_op == ">":
                        mask = df[filter_col] > comp_val
                    elif comp_op == ">=":
                        mask = df[filter_col] >= comp_val
                    elif comp_op == "<":
                        mask = df[filter_col] < comp_val
                    else:
                        mask = df[filter_col] <= comp_val
                    st.session_state.df = df[mask].copy()
                    st.session_state.filter_conditions.append(f"{filter_col} {comp_op} {comp_val}")
                    st.session_state.analysis_ready = False
                    save_analysis_state("Applied comparison filter", {"column": filter_col, "op": comp_op, "value": comp_val})
                    st.rerun()

            else:  # Exclude missing
                if st.button("Exclude missing values"):
                    st.session_state.df = df[df[filter_col].notna()].copy()
                    st.session_state.filter_conditions.append(f"{filter_col} not missing")
                    st.session_state.analysis_ready = False
                    save_analysis_state("Excluded missing", {"column": filter_col})
                    st.rerun()

        else:  # Categorical
            unique_vals = col_data.dropna().unique().tolist()
            selected_vals = st.multiselect(f"Include values for {filter_col}", unique_vals, default=unique_vals)
            if st.button("Apply category filter"):
                st.session_state.df = df[df[filter_col].isin(selected_vals)].copy()
                st.session_state.filter_conditions.append(f"{filter_col} in {selected_vals}")
                st.session_state.analysis_ready = False
                save_analysis_state("Applied category filter", {"column": filter_col, "values": selected_vals})
                st.rerun()

    # Drop columns
    st.subheader("Drop Columns")
    cols_to_drop = st.multiselect(
        "Select columns to remove",
        df.columns.tolist(),
        key="cols_to_drop",
        help="Remove columns that are not needed for analysis"
    )
    if cols_to_drop:
        if st.button("Drop selected columns"):
            st.session_state.df = df.drop(columns=cols_to_drop)
            st.session_state.filter_conditions.append(f"Dropped columns: {', '.join(cols_to_drop)}")
            st.session_state.analysis_ready = False
            _sync_data_dictionary(st.session_state.df)
            save_analysis_state("Dropped columns", {"columns": cols_to_drop})
            st.rerun()

    # Show active filters
    if st.session_state.filter_conditions:
        st.subheader("Active Filters")
        for i, cond in enumerate(st.session_state.filter_conditions):
            st.write(f"{i+1}. {cond}")

    return st.session_state.df


def variable_transformation_section(df: pd.DataFrame) -> pd.DataFrame:
    """Variable transformation section."""
    st.header("Variable Transformations")
    st.write("Transform variables for analysis (log, categorize, etc.)")

    # Show success message from previous transformation (appears inline)
    _show_inline_success()

    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    transform_type = st.selectbox(
        "Transformation type",
        ["Log transform", "Square root", "Z-score standardization",
         "Categorize continuous", "Dichotomize at threshold",
         "Combine categories", "Winsorize", "Create interaction"]
    )

    if transform_type == "Log transform":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="log_col")
        base = st.selectbox("Base", ["natural", "log10", "log2"])
        if st.button("Apply log transform"):
            with st.spinner("Applying transformation..."):
                result, new_col = apply_log_transform(df, col, base)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state(f"Log transform ({base})", {"column": col, "new_column": new_col})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Square root":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="sqrt_col")
        if st.button("Apply square root transform"):
            with st.spinner("Applying transformation..."):
                result, new_col = apply_sqrt_transform(df, col)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Square root transform", {"column": col, "new_column": new_col})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Z-score standardization":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="zscore_col")
        if st.button("Apply z-score standardization"):
            with st.spinner("Applying transformation..."):
                result, new_col = apply_zscore_transform(df, col)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Z-score standardization", {"column": col, "new_column": new_col})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Categorize continuous":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="cat_col")
        method = st.selectbox("Method", ["quantiles", "equal_width", "custom"])
        n_cats = st.number_input("Number of categories", min_value=2, max_value=10, value=4)
        custom_bins = None
        labels = None
        if method == "custom":
            bins_str = st.text_input("Bin edges (comma-separated)", "0,25,50,75,100")
            try:
                custom_bins = [float(x.strip()) for x in bins_str.split(",")]
            except ValueError:
                st.error("Invalid bin edges")
        labels_str = st.text_input("Labels (comma-separated, optional)", "")
        if labels_str.strip():
            labels = [x.strip() for x in labels_str.split(",")]
        if st.button("Apply categorization"):
            with st.spinner("Applying transformation..."):
                result, new_col = categorize_continuous(df, col, method, n_cats, custom_bins, labels)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Categorized variable", {"column": col, "method": method})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Dichotomize at threshold":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="dich_col")
        threshold = st.number_input("Threshold", value=float(df[col].median()))
        st.caption(f"Values **< {threshold}** → Lower group | Values **≥ {threshold}** → Upper group")
        label_below = st.text_input("Label for values below threshold (<)", "Low")
        label_above = st.text_input("Label for values at or above threshold (≥)", "High")
        if st.button("Apply dichotomization"):
            with st.spinner("Applying transformation..."):
                result, new_col = dichotomize(df, col, threshold, (label_below, label_above))
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Dichotomized variable", {"column": col, "threshold": threshold})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Combine categories":
        if not cat_cols:
            st.info("No categorical columns available.")
            return df
        col = st.selectbox("Column", cat_cols, key="combine_col")
        current_levels = df[col].dropna().unique().tolist()
        st.write(f"Current levels: {current_levels}")
        mapping_str = st.text_area(
            "Mapping (one per line: old_value=new_value)",
            "\n".join([f"{v}={v}" for v in current_levels])
        )
        mapping = {}
        for line in mapping_str.strip().split("\n"):
            if "=" in line:
                old, new = line.split("=", 1)
                mapping[old.strip()] = new.strip()
        if st.button("Apply recoding"):
            with st.spinner("Applying transformation..."):
                result, new_col = combine_categories(df, col, mapping)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Combined categories", {"column": col})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Winsorize":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return df
        col = st.selectbox("Column", numeric_cols, key="wins_col")
        lower = st.number_input("Lower percentile", min_value=0.0, max_value=0.5, value=0.01)
        upper = st.number_input("Upper percentile", min_value=0.5, max_value=1.0, value=0.99)
        if st.button("Apply winsorization"):
            with st.spinner("Applying transformation..."):
                result, new_col = winsorize(df, col, lower, upper)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Winsorized variable", {"column": col})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    elif transform_type == "Create interaction":
        col1 = st.selectbox("First column", cols, key="int_col1")
        col2 = st.selectbox("Second column", cols, key="int_col2")
        if st.button("Create interaction term"):
            with st.spinner("Creating interaction..."):
                result, new_col = create_interaction(df, col1, col2)
                st.session_state.df = result
                _update_data_dictionary_for_new_columns(result)
                _track_new_column(new_col)
                save_analysis_state("Created interaction", {"columns": [col1, col2]})
                _queue_toast(f"Variable '{new_col}' created successfully!")
                st.rerun()

    # Outlier detection
    st.subheader("Outlier Detection")
    if numeric_cols:
        outlier_col = st.selectbox("Check for outliers in", numeric_cols, key="outlier_col")
        outlier_method = st.selectbox("Method", ["iqr", "zscore"])
        threshold = st.number_input(
            "Threshold (IQR multiplier or z-score)",
            value=1.5 if outlier_method == "iqr" else 3.0
        )
        if st.button("Detect outliers"):
            outliers = detect_outliers(df, outlier_col, outlier_method, threshold)
            n_outliers = outliers.sum()
            st.write(f"Found {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
            if n_outliers > 0 and st.checkbox("Show outlier values"):
                _themed_dataframe(df[outliers][[outlier_col]], max_height="250px")

    return st.session_state.df


def data_preparation_summary_section(df: pd.DataFrame) -> bool:
    """Show summary of data preparation and 'Ready for Analysis' button.

    Returns True if user has clicked 'Ready for Analysis'.
    """
    st.header("Data Preparation Summary")

    # Show original vs current data info
    original_cols = st.session_state.get("original_columns", [])
    new_cols = _get_new_columns()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Columns", len(original_cols))
    with col2:
        st.metric("New Variables Created", len(new_cols))

    # Show new variables if any
    if new_cols:
        st.subheader("New Variables")
        st.write("The following variables were created through transformations:")

        # Create a summary table
        new_var_info = []
        dd = st.session_state.get("data_dictionary", {})
        for col in new_cols:
            if col in df.columns:
                var_type = dd.get(col, {}).get("type", "Unknown")
                n_unique = df[col].nunique()
                n_missing = df[col].isna().sum()
                new_var_info.append({
                    "Variable": col,
                    "Type": var_type,
                    "Unique Values": n_unique,
                    "Missing": n_missing,
                    "Sample Values": str(df[col].dropna().head(3).tolist())[:50]
                })

        if new_var_info:
            _themed_dataframe(pd.DataFrame(new_var_info), max_height="300px")
    else:
        st.info("No new variables created yet. Use the Variable Transformations section above to create derived variables.")

    # Show filter info
    filter_conditions = st.session_state.get("filter_conditions", [])
    if filter_conditions:
        st.subheader("Active Filters")
        for i, cond in enumerate(filter_conditions, 1):
            st.write(f"{i}. {cond}")
        raw_df = st.session_state.get("raw_df")
        if raw_df is not None:
            st.write(f"Rows: {len(df)} (filtered from {len(raw_df)} original)")

    st.divider()

    # Ready for Analysis button
    st.subheader("Proceed to Analysis")
    st.write("Once you're done with data filtering and transformations, click the button below to proceed to analysis.")
    st.warning("After clicking 'Ready for Analysis', go back to modify filters or transformations will reset the analysis sections.")

    if st.session_state.get("analysis_ready", False):
        st.success("Analysis sections are now available below.")
        if st.button("Reset and Modify Data Preparation", type="secondary"):
            st.session_state.analysis_ready = False
            # Clear analysis results
            for key in ["data_dictionary", "table1_pub", "table1_raw", "table2",
                       "model_results", "model_compare", "table1_exposure"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return True
    else:
        if st.button("Ready for Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_ready = True
            # Sync data dictionary one final time
            _sync_data_dictionary(df)
            save_analysis_state("Ready for analysis", {
                "original_columns": len(original_cols),
                "new_columns": len(new_cols),
                "total_rows": len(df)
            })
            st.rerun()
        return False


def data_dictionary_section(df: pd.DataFrame) -> None:
    """Data dictionary configuration section."""
    st.write("Define each variable's role and type before analysis.")
    current_df = st.session_state.get("df", df)

    if "data_dictionary" not in st.session_state:
        inferred = infer_variable_types(df)
        st.session_state.data_dictionary = {
            col: {
                "role": "Ignore" if col in detect_phi_like_columns(df) else "Baseline covariate",
                "type": inferred.get(col, "Categorical"),
                "label": col,
                "units": "",
                "order": "",
                "ref_level": "",
            }
            for col in df.columns
        }

    # Sync data dictionary with current DataFrame (add new, remove stale)
    _sync_data_dictionary(current_df)

    dd = st.session_state.data_dictionary

    for col in current_df.columns:
        with st.expander(f"{col}"):
            dd[col]["role"] = st.selectbox(
                f"Role for {col}", ROLE_OPTIONS,
                index=ROLE_OPTIONS.index(dd[col]["role"]),
                key=f"role_{col}"
            )
            dd[col]["type"] = st.selectbox(
                f"Type for {col}", TYPE_OPTIONS,
                index=TYPE_OPTIONS.index(dd[col]["type"]),
                key=f"type_{col}"
            )
            dd[col]["label"] = st.text_input(f"Display label for {col}", value=dd[col]["label"], key=f"label_{col}")
            dd[col]["units"] = st.text_input(f"Units for {col}", value=dd[col]["units"], key=f"units_{col}")
            if dd[col]["type"] == "Ordinal":
                dd[col]["order"] = st.text_input(
                    f"Order for {col} (comma-separated, optional)",
                    value=dd[col].get("order", ""),
                    key=f"order_{col}"
                )
            if dd[col]["type"] in {"Binary", "Categorical", "Ordinal"} and dd[col]["role"] not in {"Ignore", "Identifier"}:
                levels = current_df[col].dropna().astype(str).unique().tolist()
                if levels:
                    ref = st.selectbox(
                        f"Reference level for {col} (models)",
                        levels,
                        index=levels.index(dd[col]["ref_level"]) if dd[col].get("ref_level") in levels else 0,
                        key=f"ref_{col}"
                    )
                    dd[col]["ref_level"] = ref

    st.session_state.data_dictionary = dd

    if st.button("Download Data Dictionary (CSV)"):
        dd_df = pd.DataFrame.from_dict(dd, orient="index").reset_index().rename(columns={"index": "variable"})
        st.download_button(
            "Download Data Dictionary CSV",
            data=dd_df.to_csv(index=False).encode("utf-8"),
            file_name="data_dictionary.csv",
            mime="text/csv",
        )

    save_analysis_state("Data dictionary updated")
    st.session_state.progress["dictionary"] = True


def table1_section(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Table 1 generation section."""
    st.header("Table 1 (Baseline Characteristics)")
    st.write("Summarize baseline characteristics by exposure group.")

    # Sync data dictionary with current DataFrame
    _sync_data_dictionary(df)

    exposure_candidates = _get_role_vars("Exposure", df)
    if not exposure_candidates:
        st.info("Define at least one Exposure in the data dictionary to proceed.")
        return None

    exposure = st.selectbox("Exposure (grouping variable)", exposure_candidates)
    exposure_levels = df[exposure].dropna().unique().tolist()
    if exposure_levels:
        ref_level = st.selectbox("Reference level for exposure (used in models)", exposure_levels)
        st.session_state.exposure_reference = ref_level

    baseline_vars = _get_role_vars("Baseline covariate", df)
    selected_vars = st.multiselect("Baseline variables", baseline_vars, default=baseline_vars)

    continuous_summary = st.radio("Continuous summary", ["Mean (SD)", "Median (IQR)"])
    nonnormal = st.multiselect(
        "Mark non-normal continuous variables",
        [v for v in selected_vars if st.session_state.data_dictionary[v]["type"] == "Continuous"],
    )

    st.write("Missing data options")
    show_missing = st.checkbox("Show missing n (%)", value=True)
    missing_as_category = st.checkbox("Treat missing as category (categoricals)", value=False)

    st.subheader("Formatting")
    dec_cont = st.number_input("Continuous decimals", min_value=0, max_value=4, value=2)
    dec_pct = st.number_input("Percent decimals", min_value=0, max_value=2, value=1)
    show_levels = st.radio("Categorical display", ["Show levels", "Top-level only"], horizontal=True)
    use_tableone = st.checkbox("Use tableone engine (recommended)", value=True)

    if st.button("Generate Table 1"):
        try:
            with st.spinner("Generating Table 1..."):
                pub_table, raw_table = generate_table1(
                    df=df,
                    exposure=exposure,
                    selected_vars=selected_vars,
                    continuous_summary=continuous_summary,
                    nonnormal=nonnormal,
                    show_missing=show_missing,
                    missing_as_category=missing_as_category,
                    dec_cont=dec_cont,
                    dec_pct=dec_pct,
                    p_threshold=0.0001,
                    show_levels=show_levels,
                    use_tableone=use_tableone,
                    data_dictionary=st.session_state.data_dictionary,
                )

            st.session_state.table1_pub = pub_table
            st.session_state.table1_raw = raw_table
            st.session_state.table1_exposure = exposure
            st.session_state.table1_n = len(df)
            st.session_state.progress["table1"] = True
            pub_table.attrs["title"] = f"Table 1 — Exposure: {exposure}"

            save_analysis_state("Generated Table 1", {"exposure": exposure, "vars": selected_vars})
            st.success("Table 1 generated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to generate Table 1: {exc}")

    if "table1_pub" in st.session_state:
        view_tabs = st.tabs(["View", "Export"])
        with view_tabs[0]:
            _themed_dataframe(st.session_state.table1_pub)
            st.info(f"Summary: {len(st.session_state.table1_pub)} rows, exposure = {st.session_state.table1_exposure}.")
        with view_tabs[1]:
            pub_table = st.session_state.table1_pub
            raw_table = st.session_state.get("table1_raw")
            st.download_button(
                "Download Table 1 as CSV",
                data=pub_table.to_csv(index=False).encode("utf-8"),
                file_name="table1.csv",
                mime="text/csv",
            )
            sheets = {"Table1 (Publication)": pub_table}
            if raw_table is not None and not raw_table.empty:
                sheets["Table1 (Raw/Verbose)"] = raw_table
            if "data_dictionary" in st.session_state:
                dd_df = pd.DataFrame.from_dict(st.session_state.data_dictionary, orient="index").reset_index().rename(columns={"index": "variable"})
                sheets["Data Dictionary"] = dd_df
            excel_bytes = to_excel_bytes(sheets)
            st.download_button(
                "Download Table 1 as Excel",
                data=excel_bytes,
                file_name="table1.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    return st.session_state.get("table1_pub")


def outcome_and_table2_section(df: pd.DataFrame) -> None:
    """Outcomes and Table 2 section."""
    st.header("Outcomes & Table 2")
    st.write("Select outcomes and compute crude associations by exposure group.")

    # Sync data dictionary with current DataFrame
    _sync_data_dictionary(df)

    exposure = st.session_state.get("table1_exposure")
    if not exposure:
        st.info("Generate Table 1 and select an exposure variable first.")
        return

    outcomes = _get_role_vars("Outcome", df)
    if not outcomes:
        st.info("Define at least one Outcome in the data dictionary.")
        return

    selected_outcomes = st.multiselect("Outcome variables", outcomes, default=outcomes[:1])
    if not selected_outcomes:
        return

    outcome_types: Dict[str, str] = {}
    for outcome in selected_outcomes:
        inferred = st.session_state.data_dictionary[outcome]["type"]
        default_type = "Binary" if inferred == "Binary" else "Continuous"
        outcome_types[outcome] = st.selectbox(
            f"Outcome type for {outcome}",
            ["Binary", "Continuous", "Time-to-event"],
            index=["Binary", "Continuous", "Time-to-event"].index(default_type),
            key=f"t2_outcome_type_{outcome}",
        )
        if outcome_types[outcome] == "Time-to-event":
            cols = df.columns.tolist()
            time_default = f"{outcome}_time" if f"{outcome}_time" in cols else cols[0]
            event_default = f"{outcome}_event" if f"{outcome}_event" in cols else cols[0]
            time_col = st.selectbox(f"Time column for {outcome}", cols, index=cols.index(time_default), key=f"t2_time_{outcome}")
            event_col = st.selectbox(f"Event column for {outcome}", cols, index=cols.index(event_default), key=f"t2_event_{outcome}")
            series = pd.to_numeric(df[time_col], errors="coerce").dropna()
            min_t = float(series.min()) if not series.empty else 0.0
            max_t = float(series.max()) if not series.empty else 1.0
            col_a, col_b = st.columns(2)
            with col_a:
                interval_start = st.number_input(f"Start time for p-value ({outcome})", value=min_t, key=f"t2_start_{outcome}")
            with col_b:
                interval_end = st.number_input(f"End time for p-value ({outcome})", value=max_t, key=f"t2_end_{outcome}")
            st.session_state.time_event_cols[outcome] = {
                "time": time_col,
                "event": event_col,
                "interval": (interval_start, interval_end),
            }

    st.session_state.selected_outcomes = selected_outcomes
    st.session_state.selected_outcome_types = outcome_types

    use_tableone = st.checkbox("Use tableone engine for Table 2 summary", value=True)
    create_or_table = st.checkbox("Also create OR table (binary outcomes only)", value=True)
    cont_summary = st.radio("Continuous summary (Table 2)", ["Mean (SD)", "Median (IQR)"], horizontal=True)
    cont_dec = st.number_input("Continuous decimals (Table 2)", min_value=0, max_value=4, value=2)
    pct_dec = st.number_input("Percent decimals (Table 2)", min_value=0, max_value=2, value=1)

    if st.button("Generate Table 2"):
        with st.spinner("Generating Table 2..."):
            table2, table2_raw = generate_table2(
                df=df,
                exposure=exposure,
                outcomes=selected_outcomes,
                outcome_types=outcome_types,
                use_tableone=use_tableone,
                time_event_cols=st.session_state.get("time_event_cols", {}),
                cont_summary=cont_summary,
                cont_dec=cont_dec,
                pct_dec=pct_dec,
                data_dictionary=st.session_state.data_dictionary,
            )
            st.session_state.table2 = table2
            st.session_state.table2_raw = table2_raw
            st.session_state.table2_n = len(df)
            st.session_state.progress["table2"] = True
            table2.attrs["title"] = f"Table 2 — Exposure: {exposure}"

            if create_or_table:
                or_table = generate_or_table(
                    df, exposure, selected_outcomes, outcome_types,
                    st.session_state.data_dictionary,
                    st.session_state.get("exposure_reference")
                )
                st.session_state.table2_or = or_table

        save_analysis_state("Generated Table 2", {"outcomes": selected_outcomes})
        st.success("Table 2 generated.")
        st.rerun()

    if "table2" in st.session_state:
        view_tabs = st.tabs(["View", "Export"])
        with view_tabs[0]:
            _themed_dataframe(st.session_state.table2)
            st.info(f"Summary: {len(st.session_state.table2)} rows.")
            if "table2_or" in st.session_state and not st.session_state.table2_or.empty:
                st.subheader("Crude OR Table (Binary Outcomes)")
                _themed_dataframe(st.session_state.table2_or)

            if LIFELINES_AVAILABLE:
                st.subheader("Kaplan-Meier Curves (Time-to-event)")
                for outcome in st.session_state.get("selected_outcomes", []):
                    if st.session_state.get("selected_outcome_types", {}).get(outcome) != "Time-to-event":
                        continue
                    fig = create_km_plot(df, exposure, outcome, st.session_state.get("time_event_cols", {}))
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.info(f"KM curve not available for {outcome}.")

        with view_tabs[1]:
            t2 = st.session_state.table2
            st.download_button("Download Table 2 as CSV", data=t2.to_csv(index=False).encode("utf-8"), file_name="table2.csv")
            sheets = {"Table2": t2}
            if "table2_or" in st.session_state and not st.session_state.table2_or.empty:
                sheets["Table2 OR"] = st.session_state.table2_or
            excel_bytes = to_excel_bytes(sheets)
            st.download_button("Download Table 2 as Excel", data=excel_bytes, file_name="table2.xlsx")


def graphing_section() -> None:
    """Graphing and plotting section."""
    st.header("Graphing and Plots")
    df = st.session_state.get("df")
    if df is None:
        st.info("Load data first.")
        return

    chart = st.selectbox("Chart type", ["Scatter", "Bar", "Histogram", "Box"])
    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    title = st.text_input("Plot title (optional)", "")
    x_label = st.text_input("X-axis label (optional)", "")
    y_label = st.text_input("Y-axis label (optional)", "")

    fig = None

    if chart == "Scatter":
        x = st.selectbox("X", cols, key="scatter_x")
        y = st.selectbox("Y", cols, key="scatter_y")
        hue = st.selectbox("Hue (optional)", ["None"] + cols, key="scatter_hue")
        if st.button("Generate scatter plot"):
            fig = create_scatter(df, x, y, hue if hue != "None" else None, title, x_label, y_label)

    elif chart == "Bar":
        x = st.selectbox("X (category)", cols, key="bar_x")
        y = st.selectbox("Y (numeric)", numeric_cols if numeric_cols else cols, key="bar_y")
        if st.button("Generate bar chart"):
            fig, ax = plt.subplots(figsize=(10, 6))
            means = df.groupby(x)[y].mean()
            ax.bar(means.index.astype(str), means.values, color="#4B9EFE")
            ax.set_title(title or f"Mean {y} by {x}")
            ax.set_xlabel(x_label or x)
            ax.set_ylabel(y_label or f"Mean {y}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

    elif chart == "Histogram":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        x = st.selectbox("Variable", numeric_cols, key="hist_x")
        bins = st.slider("Bins", 5, 50, 20)
        group = st.selectbox("Group by (optional)", ["None"] + cols, key="hist_group")
        if st.button("Generate histogram"):
            fig = create_histogram(df, x, bins, group if group != "None" else None, title, x_label)

    elif chart == "Box":
        if not numeric_cols:
            st.info("No numeric columns available.")
            return
        y = st.selectbox("Y (numeric)", numeric_cols, key="box_y")
        group = st.selectbox("Group by (optional)", ["None"] + cols, key="box_group")
        if st.button("Generate box plot"):
            fig = create_boxplot(df, y, group if group != "None" else None, title, x_label, y_label)

    if fig is not None:
        st.pyplot(fig)
        buf = fig_to_bytes(fig, dpi=600)
        st.download_button("Export Graph (PNG)", data=buf, file_name="graph.png", mime="image/png")


def covariate_suggestion_section(df: pd.DataFrame) -> None:
    """Covariate selection section."""
    st.header("Covariate Selection")
    st.write("Suggested covariates based on baseline status, balance, and missingness.")

    # Sync data dictionary with current DataFrame
    _sync_data_dictionary(df)

    exposure = st.session_state.get("table1_exposure")
    if not exposure:
        st.info("Select an exposure variable in Table 1 first.")
        return

    baseline_vars = _get_role_vars("Baseline covariate", df)
    suggested = []
    for var in baseline_vars:
        miss = df[var].isna().mean()
        if miss < 0.3:
            smd = compute_smd(df, exposure, var, st.session_state.data_dictionary[var]["type"])
            if smd is not None and abs(smd) > 0.1:
                suggested.append(var)
    for key in ["age", "sex", "gender"]:
        for var in baseline_vars:
            if key in var.lower() and var not in suggested:
                suggested.append(var)

    st.session_state.suggested_covariates = suggested
    covariates = st.multiselect("Select covariates for adjustment", baseline_vars, default=suggested)
    st.session_state.selected_covariates = covariates

    st.subheader("Do not adjust for")
    st.write(", ".join(_get_role_vars("Post-exposure", df) + _get_role_vars("Outcome", df)))

    st.expander("Confounding checklist").write(
        "- Is this variable measured before exposure?\n"
        "- Could it be affected by the exposure?\n"
        "- Is it a proxy for disease severity?"
    )


def regression_section(df: pd.DataFrame) -> None:
    """Multivariable regression section."""
    st.header("Multivariable Regression")

    # Sync data dictionary with current DataFrame
    _sync_data_dictionary(df)

    exposure = st.session_state.get("table1_exposure")
    outcomes = _get_role_vars("Outcome", df)
    if not exposure or not outcomes:
        st.info("Select exposure and outcome variables first.")
        return

    outcome = st.selectbox("Outcome", outcomes)
    outcome_type = st.selectbox("Outcome type", ["Binary", "Continuous", "Time-to-event"])
    covariates = st.session_state.get("selected_covariates", [])
    missing_strategy = st.radio("Missing data handling", ["Complete-case", "Simple imputation"])
    st.session_state.missing_strategy = missing_strategy

    bootstrap = st.checkbox("Compute bootstrap 95% CI (slow)", value=False)
    n_boot = st.number_input("Bootstrap samples", min_value=50, max_value=1000, value=200) if bootstrap else 0

    st.session_state.current_outcome = outcome
    st.session_state.current_outcome_type = outcome_type

    st.subheader("Model readiness")
    readiness_notes = assumption_warnings(
        df, outcome, exposure, covariates, outcome_type,
        st.session_state.get("time_event_cols")
    )
    miss = summarize_missingness(df)
    high_miss = miss[miss["Percent Missing"] > 10].index.tolist()
    st.write(f"High missingness (>10%): {high_miss if high_miss else 'None'}")
    for note in readiness_notes:
        st.warning(note)

    if outcome_type == "Time-to-event":
        cols = df.columns.tolist()
        time_default = f"{outcome}_time" if f"{outcome}_time" in cols else cols[0]
        event_default = f"{outcome}_event" if f"{outcome}_event" in cols else cols[0]
        time_col = st.selectbox("Time column", cols, index=cols.index(time_default))
        event_col = st.selectbox("Event column", cols, index=cols.index(event_default))
        st.session_state.time_event_cols[outcome] = {"time": time_col, "event": event_col}

    if st.button("Fit adjusted model"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        results, note, n_used = fit_model(
            df, outcome, exposure, covariates, outcome_type, missing_strategy,
            st.session_state.data_dictionary,
            st.session_state.get("exposure_reference"),
            st.session_state.get("time_event_cols"),
            progress_callback=update_progress,
        )
        st.session_state.model_results = results
        st.session_state.model_note = note
        st.session_state.model_n = n_used
        st.session_state.progress["regression"] = True

        # Crude model comparison
        if outcome_type in {"Binary", "Continuous"} and not results.empty:
            crude, _, _ = fit_model(
                df, outcome, exposure, [], outcome_type, missing_strategy,
                st.session_state.data_dictionary,
                st.session_state.get("exposure_reference"),
                st.session_state.get("time_event_cols"),
            )
            if not crude.empty:
                merged = crude.merge(results, on="Variable", how="outer", suffixes=(" (Crude)", " (Adjusted)"))
                st.session_state.model_compare = merged

        # Bootstrap CI
        if bootstrap and outcome_type in {"Binary", "Continuous"}:
            boot_results = bootstrap_ci(
                df, outcome, exposure, covariates, outcome_type, n_boot,
                st.session_state.data_dictionary,
                st.session_state.get("exposure_reference"),
                progress_callback=update_progress,
            )
            st.session_state.bootstrap_ci = boot_results

        save_analysis_state("Fit regression model", {"outcome": outcome, "type": outcome_type})
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        st.success("Model fit complete.")
        st.rerun()

    if "model_results" in st.session_state:
        view_tabs = st.tabs(["View", "Forest Plot", "Diagnostics", "Export"])

        with view_tabs[0]:
            results = st.session_state.model_results
            _themed_dataframe(results)
            st.caption(st.session_state.get("model_note", ""))
            st.info(f"n used = {st.session_state.get('model_n', '—')}")

            if "model_compare" in st.session_state:
                st.subheader("Crude vs Adjusted")
                _themed_dataframe(st.session_state.model_compare)

        with view_tabs[1]:
            results = st.session_state.model_results
            if not results.empty:
                # Determine effect column
                if "OR" in results.columns:
                    effect_col = "OR"
                    xlabel = "Odds Ratio (95% CI)"
                    null_val = 1.0
                    log_scale = True
                elif "HR" in results.columns:
                    effect_col = "HR"
                    xlabel = "Hazard Ratio (95% CI)"
                    null_val = 1.0
                    log_scale = True
                else:
                    effect_col = "Beta"
                    xlabel = "Coefficient (95% CI)"
                    null_val = 0.0
                    log_scale = False

                forest_fig = create_forest_plot(
                    results,
                    effect_col=effect_col,
                    ci_lower_col="CI Lower",
                    ci_upper_col="CI Upper",
                    label_col="Variable",
                    title="Forest Plot",
                    xlabel=xlabel,
                    null_value=null_val,
                    log_scale=log_scale,
                )
                if forest_fig:
                    st.pyplot(forest_fig)
                    buf = fig_to_bytes(forest_fig, dpi=600)
                    st.download_button("Export Forest Plot (PNG)", data=buf, file_name="forest_plot.png", mime="image/png")
                else:
                    st.info("Unable to create forest plot from results.")

        with view_tabs[2]:
            outcome_type = st.session_state.get("current_outcome_type", "Continuous")
            covariates = st.session_state.get("selected_covariates", [])
            if outcome_type in {"Binary", "Continuous"} and covariates:
                vif_df = compute_vif(df, covariates)
                if not vif_df.empty:
                    st.subheader("Variance Inflation Factors")
                    _themed_dataframe(vif_df)
                    st.caption("High VIF (>5-10) suggests multicollinearity.")

            if "bootstrap_ci" in st.session_state and not st.session_state.bootstrap_ci.empty:
                st.subheader("Bootstrap 95% CI")
                _themed_dataframe(st.session_state.bootstrap_ci)

        with view_tabs[3]:
            model_df = st.session_state.model_results
            st.download_button(
                "Download model results (CSV)",
                data=model_df.to_csv(index=False).encode("utf-8"),
                file_name="model_results.csv"
            )
            st.download_button(
                "Download model results (Excel)",
                data=to_excel_bytes({"Regression": model_df}),
                file_name="model_results.xlsx"
            )


def subgroup_analysis_section(df: pd.DataFrame) -> None:
    """Subgroup analysis section."""
    st.header("Subgroup Analysis")

    # Sync data dictionary with current DataFrame
    _sync_data_dictionary(df)

    exposure = st.session_state.get("table1_exposure")
    outcomes = _get_role_vars("Outcome", df)
    if not exposure or not outcomes:
        st.info("Select exposure and outcome variables first.")
        return

    strat_var = st.selectbox("Stratify by", [c for c in df.columns if c != exposure])
    selected_outcomes = st.multiselect("Outcomes", outcomes, default=outcomes[:1])
    if not selected_outcomes:
        return

    outcome_types: Dict[str, str] = {}
    for outcome in selected_outcomes:
        inferred = st.session_state.data_dictionary[outcome]["type"]
        default_type = "Binary" if inferred == "Binary" else "Continuous"
        outcome_types[outcome] = st.selectbox(
            f"Outcome type for {outcome}",
            ["Binary", "Continuous", "Time-to-event"],
            index=["Binary", "Continuous", "Time-to-event"].index(default_type),
            key=f"subgroup_type_{outcome}",
        )

    use_tableone = st.checkbox("Use tableone engine (stratified)", value=True)
    cont_summary = st.radio("Continuous summary", ["Mean (SD)", "Median (IQR)"], horizontal=True)
    cont_dec = st.number_input("Continuous decimals", min_value=0, max_value=4, value=2, key="subgroup_cont_dec")
    pct_dec = st.number_input("Percent decimals", min_value=0, max_value=2, value=1, key="subgroup_pct_dec")
    st.warning("Stratified analyses can be underpowered; interpret cautiously.")

    if st.button("Generate stratified Table 2"):
        progress_bar = st.progress(0)
        strat_tables = {}
        levels = df[strat_var].dropna().unique()
        for i, level in enumerate(levels):
            progress_bar.progress((i + 1) / len(levels))
            subset = df[df[strat_var] == level]
            table2, _ = generate_table2(
                df=subset,
                exposure=exposure,
                outcomes=selected_outcomes,
                outcome_types=outcome_types,
                use_tableone=use_tableone,
                time_event_cols=st.session_state.get("time_event_cols", {}),
                cont_summary=cont_summary,
                cont_dec=cont_dec,
                pct_dec=pct_dec,
                data_dictionary=st.session_state.data_dictionary,
            )
            strat_tables[str(level)] = table2
        st.session_state.stratified_table2 = strat_tables
        save_analysis_state("Generated stratified Table 2", {"strat_var": strat_var})
        st.success("Stratified Table 2 generated.")

    if "stratified_table2" in st.session_state:
        for level, table in st.session_state.stratified_table2.items():
            st.subheader(f"Stratum: {strat_var} = {level}")
            _themed_dataframe(table)
        excel_bytes = to_excel_bytes({f"Table2_{lvl}": tbl for lvl, tbl in st.session_state.stratified_table2.items()})
        st.download_button("Download stratified Table 2 (Excel)", data=excel_bytes, file_name="table2_stratified.xlsx")


def results_summary_dashboard(df: pd.DataFrame) -> None:
    """Results Summary Dashboard showing all key results in one view."""
    st.header("Results Summary Dashboard")
    st.write("Overview of your analysis results in one place.")

    # Check if we have any results to show
    has_table1 = "table1_pub" in st.session_state
    has_table2 = "table2" in st.session_state
    has_regression = "model_results" in st.session_state

    if not any([has_table1, has_table2, has_regression]):
        st.info("No results to display yet. Complete the analysis sections to see your results summary here.")
        return

    # Study Parameters Card
    st.subheader("Study Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)

    exposure = st.session_state.get("table1_exposure", "Not set")
    outcomes = _get_role_vars("Outcome", df)
    covariates = st.session_state.get("selected_covariates", [])

    with param_col1:
        st.metric("Exposure", exposure if exposure else "Not set")
        if exposure and exposure in df.columns:
            n_exposed = df[exposure].value_counts()
            if len(n_exposed) > 0:
                st.caption(f"Groups: {', '.join([f'{k}: n={v}' for k, v in n_exposed.head(3).items()])}")

    with param_col2:
        st.metric("Outcomes", len(outcomes))
        if outcomes:
            st.caption(", ".join(outcomes[:3]) + ("..." if len(outcomes) > 3 else ""))

    with param_col3:
        st.metric("Covariates", len(covariates))
        if covariates:
            st.caption(", ".join(covariates[:3]) + ("..." if len(covariates) > 3 else ""))

    st.divider()

    # Key Results in columns
    results_col1, results_col2 = st.columns(2)

    with results_col1:
        # Table 1 Summary
        if has_table1:
            st.subheader("Table 1 Highlights")
            table1 = st.session_state.table1_pub

            # Show sample size
            n_total = st.session_state.get("table1_n", len(df))
            st.write(f"**Total N:** {n_total}")

            # Show the table (condensed)
            with st.expander("View Table 1", expanded=False):
                _themed_dataframe(table1, max_height="300px")

            # Highlight variables with high SMD if available
            if "table1_raw" in st.session_state:
                raw_table = st.session_state.table1_raw
                if "SMD" in raw_table.columns:
                    try:
                        smd_values = pd.to_numeric(raw_table["SMD"], errors="coerce")
                        high_smd = raw_table[smd_values.abs() > 0.1]
                        if not high_smd.empty:
                            st.warning(f"**{len(high_smd)} variables with SMD > 0.1** (potential imbalance)")
                    except Exception:
                        pass
        else:
            st.subheader("Table 1")
            st.info("Generate Table 1 in the Tables tab to see summary here.")

    with results_col2:
        # Main Effect Estimates
        if has_regression:
            st.subheader("Main Effect Estimates")
            model_results = st.session_state.model_results

            # Find the exposure effect (usually the first non-intercept row)
            exposure_var = st.session_state.get("table1_exposure")
            if exposure_var and "Variable" in model_results.columns:
                exposure_rows = model_results[model_results["Variable"].str.contains(exposure_var, na=False, case=False)]
                if not exposure_rows.empty:
                    for _, row in exposure_rows.iterrows():
                        effect_col = None
                        for col in ["OR", "HR", "Coefficient"]:
                            if col in row and pd.notna(row[col]):
                                effect_col = col
                                break

                        if effect_col:
                            effect = row[effect_col]
                            ci_lower = row.get("CI Lower", "")
                            ci_upper = row.get("CI Upper", "")
                            pval = row.get("P-value", "")

                            st.metric(
                                f"{row['Variable']}",
                                f"{effect_col}: {effect:.2f}" if isinstance(effect, (int, float)) else f"{effect_col}: {effect}",
                                f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]" if isinstance(ci_lower, (int, float)) else ""
                            )
                            if pval:
                                pval_formatted = f"{pval:.4f}" if isinstance(pval, (int, float)) else pval
                                st.caption(f"P-value: {pval_formatted}")

            # Show full results in expander
            with st.expander("View Full Regression Results", expanded=False):
                _themed_dataframe(model_results, max_height="300px")
        else:
            st.subheader("Regression Results")
            st.info("Fit a regression model in the Regression tab to see results here.")

    st.divider()

    # Forest Plot (full width)
    if has_regression:
        st.subheader("Forest Plot")
        model_results = st.session_state.model_results

        # Determine effect type
        outcome_type = st.session_state.get("current_outcome_type", "Continuous")
        if outcome_type == "Binary":
            effect_col, xlabel, null_val, log_scale = "OR", "Odds Ratio", 1.0, True
        elif outcome_type == "Time-to-event":
            effect_col, xlabel, null_val, log_scale = "HR", "Hazard Ratio", 1.0, True
        else:
            effect_col, xlabel, null_val, log_scale = "Coefficient", "Coefficient", 0.0, False

        if effect_col in model_results.columns:
            forest_fig = create_forest_plot(
                model_results,
                effect_col=effect_col,
                ci_lower_col="CI Lower",
                ci_upper_col="CI Upper",
                label_col="Variable",
                title="Effect Estimates with 95% CI",
                xlabel=xlabel,
                null_value=null_val,
                log_scale=log_scale,
            )
            if forest_fig:
                st.pyplot(forest_fig)
            else:
                st.info("Unable to generate forest plot.")
        else:
            st.info(f"Effect column '{effect_col}' not found in results.")

    st.divider()

    # Table 2 Summary
    if has_table2:
        st.subheader("Outcome Summary (Table 2)")
        table2 = st.session_state.table2
        with st.expander("View Table 2", expanded=False):
            _themed_dataframe(table2, max_height="300px")

    # Quick Export
    st.subheader("Quick Export")
    export_col1, export_col2, export_col3 = st.columns(3)

    sheets = {}
    if has_table1:
        sheets["Table1"] = st.session_state.table1_pub
    if has_table2:
        sheets["Table2"] = st.session_state.table2
    if has_regression:
        sheets["Regression"] = round_numeric_columns(clean_regression_results(st.session_state.model_results))

    with export_col1:
        if sheets:
            bundle = to_excel_bytes(sheets)
            st.download_button(
                "Download All Results (Excel)",
                data=bundle,
                file_name="results_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with export_col2:
        if has_regression:
            # Export forest plot
            outcome_type = st.session_state.get("current_outcome_type", "Continuous")
            if outcome_type == "Binary":
                effect_col, xlabel, null_val, log_scale = "OR", "Odds Ratio", 1.0, True
            elif outcome_type == "Time-to-event":
                effect_col, xlabel, null_val, log_scale = "HR", "Hazard Ratio", 1.0, True
            else:
                effect_col, xlabel, null_val, log_scale = "Coefficient", "Coefficient", 0.0, False

            if effect_col in st.session_state.model_results.columns:
                forest_fig = create_forest_plot(
                    st.session_state.model_results,
                    effect_col=effect_col,
                    ci_lower_col="CI Lower",
                    ci_upper_col="CI Upper",
                    label_col="Variable",
                    title="Forest Plot",
                    xlabel=xlabel,
                    null_value=null_val,
                    log_scale=log_scale,
                )
                if forest_fig:
                    buf = fig_to_bytes(forest_fig, dpi=300)
                    st.download_button(
                        "Download Forest Plot (PNG)",
                        data=buf,
                        file_name="forest_plot.png",
                        mime="image/png",
                        use_container_width=True,
                    )

    with export_col3:
        if has_table1:
            st.download_button(
                "Download Table 1 (CSV)",
                data=st.session_state.table1_pub.to_csv(index=True).encode("utf-8"),
                file_name="table1.csv",
                mime="text/csv",
                use_container_width=True,
            )


def report_and_export_section() -> None:
    """Reproducibility and export section."""
    st.header("Reproducibility & Export")

    # Sync data dictionary with current DataFrame
    df = st.session_state.get("df")
    if df is not None:
        _sync_data_dictionary(df)

    config = {
        "app_version": APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "study_design": st.session_state.get("study_design"),
        "data_dictionary": st.session_state.get("data_dictionary"),
        "exposure": st.session_state.get("table1_exposure"),
        "outcomes": _get_role_vars("Outcome"),
        "covariates": st.session_state.get("selected_covariates"),
        "missing_strategy": st.session_state.get("missing_strategy", "Complete-case"),
        "filter_conditions": st.session_state.get("filter_conditions", []),
        "analysis_log": get_history().get_analysis_log(),
    }

    run_config_json = build_run_config_json(config)
    st.subheader("Run Configuration")
    if st.checkbox("Show Run Configuration", value=False):
        st.code(run_config_json, language="json")

    st.download_button("Download Run Config JSON", data=run_config_json.encode("utf-8"), file_name="run_config.json")

    st.subheader("Download Bundle")
    sheets = {}
    if "table1_pub" in st.session_state:
        sheets["Table1"] = st.session_state.table1_pub
    if "table2" in st.session_state:
        sheets["Table2"] = st.session_state.table2
    if "model_results" in st.session_state:
        sheets["Regression"] = round_numeric_columns(clean_regression_results(st.session_state.model_results))
    if "data_dictionary" in st.session_state:
        dd = pd.DataFrame.from_dict(st.session_state.data_dictionary, orient="index").reset_index().rename(columns={"index": "variable"})
        sheets["Data Dictionary"] = dd
    sheets["Run Config"] = run_config_json
    analysis_log = get_history().get_analysis_log()
    if analysis_log:
        sheets["Analysis Log"] = pd.DataFrame(analysis_log)

    if sheets:
        bundle = to_excel_bytes(sheets)
        st.download_button(
            "Download Excel Bundle",
            data=bundle,
            file_name="analysis_bundle.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Methods (for report)")
    methods_text = st.text_area(
        "Methods text",
        value=st.session_state.get("report_methods_text", ""),
        height=150,
        key="report_methods_text",
    )

    st.subheader("Report preview")
    if "table1_pub" in st.session_state:
        st.write("Table 1 (preview)")
        _themed_dataframe(st.session_state.table1_pub, max_rows=5, max_height="200px")
    if "table2" in st.session_state:
        st.write("Table 2 (preview)")
        _themed_dataframe(st.session_state.table2, max_rows=5, max_height="200px")
    if "model_results" in st.session_state:
        st.write("Model results (preview)")
        _themed_dataframe(st.session_state.model_results, max_rows=5, max_height="200px")

    st.subheader("Generate report")
    if not DOCX_AVAILABLE:
        st.info("Install python-docx to enable Word report generation.")
        return

    if st.button("Generate Word Report"):
        report_bytes = generate_word_report(
            methods_text=methods_text,
            table1=st.session_state.get("table1_pub"),
            table2=st.session_state.get("table2"),
            table2_or=st.session_state.get("table2_or"),
            model_results=st.session_state.get("model_results"),
            model_compare=st.session_state.get("model_compare"),
            stratified_table2=st.session_state.get("stratified_table2"),
            custom_summary_table=st.session_state.get("custom_summary_table"),
        )
        if report_bytes:
            st.download_button("Download Word report", data=report_bytes, file_name="analysis_report.docx")


def custom_table_builder_section(df: pd.DataFrame) -> None:
    """Custom summary table builder section."""
    st.header("Custom Summary Table")
    st.write("Build a custom summary table with selected variables and statistics.")

    cols = df.columns.tolist()
    selected = st.multiselect("Variables", cols)
    if not selected:
        st.info("Select at least one variable.")
        return

    stats_options = st.multiselect(
        "Statistics",
        ["Mean", "SD", "Median", "IQR", "Min", "Max", "Count", "Percent"],
        default=["Mean", "SD", "Median", "IQR", "Count"],
    )
    group_col = st.selectbox("Group by (optional)", ["None"] + cols, key="custom_table_group")
    dec = st.number_input("Decimals", min_value=0, max_value=4, value=2)

    rows = []
    if group_col == "None":
        for col in selected:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                row = {"Variable": col}
                if "Count" in stats_options:
                    row["Count"] = series.notna().sum()
                if "Mean" in stats_options:
                    row["Mean"] = round(series.mean(), dec)
                if "SD" in stats_options:
                    row["SD"] = round(series.std(ddof=1), dec)
                if "Median" in stats_options:
                    row["Median"] = round(series.median(), dec)
                if "IQR" in stats_options:
                    row["IQR"] = f"{round(series.quantile(0.25), dec)}–{round(series.quantile(0.75), dec)}"
                if "Min" in stats_options:
                    row["Min"] = round(series.min(), dec)
                if "Max" in stats_options:
                    row["Max"] = round(series.max(), dec)
                rows.append(row)
            else:
                counts = series.value_counts(dropna=True)
                total = len(df)
                for level, cnt in counts.items():
                    row = {"Variable": f"{col}: {level}"}
                    if "Count" in stats_options:
                        row["Count"] = int(cnt)
                    if "Percent" in stats_options:
                        row["Percent"] = f"{(cnt / total * 100) if total else 0:.{dec}f}%"
                    rows.append(row)
    else:
        for col in selected:
            for level, sub in df.groupby(group_col):
                series = sub[col]
                row = {"Group": level, "Variable": col}
                if pd.api.types.is_numeric_dtype(series):
                    if "Count" in stats_options:
                        row["Count"] = series.notna().sum()
                    if "Mean" in stats_options:
                        row["Mean"] = round(series.mean(), dec)
                    if "SD" in stats_options:
                        row["SD"] = round(series.std(ddof=1), dec)
                    if "Median" in stats_options:
                        row["Median"] = round(series.median(), dec)
                    if "IQR" in stats_options:
                        row["IQR"] = f"{round(series.quantile(0.25), dec)}–{round(series.quantile(0.75), dec)}"
                    if "Min" in stats_options:
                        row["Min"] = round(series.min(), dec)
                    if "Max" in stats_options:
                        row["Max"] = round(series.max(), dec)
                    rows.append(row)
                else:
                    counts = series.value_counts(dropna=True)
                    total = len(sub)
                    for cat, cnt in counts.items():
                        row = {"Group": level, "Variable": f"{col}: {cat}"}
                        if "Count" in stats_options:
                            row["Count"] = int(cnt)
                        if "Percent" in stats_options:
                            row["Percent"] = f"{(cnt / total * 100) if total else 0:.{dec}f}%"
                        rows.append(row)

    table = pd.DataFrame(rows)
    st.session_state.custom_summary_table = table
    _themed_dataframe(table)
    st.download_button(
        "Download Custom Summary (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="custom_summary.csv",
    )


# -----------------------------
# Main Application
# -----------------------------

def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Clinical Data Analysis", page_icon="S4S_emblem.png", layout="wide")
    

    
    # Initialize session state first (needed for theme)
    _init_session_state()

    # Inject theme-based CSS
    st.markdown(_get_theme_css(), unsafe_allow_html=True)

    # Theme colors for header
    is_dark = st.session_state.get("theme", "light") == "dark"
    header_bg = "#1e293b" if is_dark else "#ffffff"
    header_border = "#334155" if is_dark else "#e5e7eb"
    header_text = "#f1f5f9" if is_dark else "#0f172a"
    header_subtext = "#94a3b8" if is_dark else "#6b7280"

    # Theme toggle at top of sidebar (most visible location)
    toggle_bg = "#334155" if is_dark else "#f1f5f9"
    toggle_border = "#475569" if is_dark else "#e2e8f0"
    toggle_text = "#f1f5f9" if is_dark else "#0f172a"

    st.sidebar.markdown(
        f"""
        <div style="background:{toggle_bg};border:1px solid {toggle_border};border-radius:10px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:14px;font-weight:600;color:{toggle_text};">
                {"🌙 Dark Mode" if is_dark else "☀️ Light Mode"}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    new_theme = st.sidebar.toggle(
        "Switch Theme",
        value=is_dark,
        key="theme_toggle",
        label_visibility="collapsed"
    )
    if new_theme != is_dark:
        st.session_state.theme = "dark" if new_theme else "light"
        st.rerun()

    st.sidebar.markdown("---")

    # Header with emblem
    emblem_svg = '''<svg width="60" height="32" viewBox="0 0 119 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M56.5 27.6699C59.8333 29.5944 59.8333 34.4056 56.5 36.3301L23.5 55.3827C20.1667 57.3072 16 54.9016 16 51.0526L16 12.9474C16 9.09843 20.1667 6.69281 23.5 8.61731L56.5 27.6699Z" fill="#3C90FF"/>
<path d="M74.5 27.6699C77.8333 29.5944 77.8333 34.4056 74.5 36.3301L41.5 55.3827C38.1667 57.3072 34 54.9016 34 51.0526L34 12.9474C34 9.09843 38.1667 6.69281 41.5 8.61731L74.5 27.6699Z" fill="#F96900"/>
<path d="M87.0001 26.8038C91.0001 29.1132 91.0001 34.887 87.0001 37.1964L58.5001 53.6505C56.5001 54.8051 54.0002 53.3621 54.0001 51.0528L54.0001 12.9473C54.0001 10.638 56.5001 9.195 58.5001 10.3497L87.0001 26.8038Z" stroke="black" stroke-width="4"/>
<path d="M84 26.8038C80 29.1132 80 34.887 84 37.1964L112.5 53.6505C114.5 54.8051 117 53.3621 117 51.0528L117 12.9473C117 10.638 114.5 9.195 112.5 10.3497L84 26.8038Z" stroke="black" stroke-width="4"/>
<path d="M80 54H92" stroke="black" stroke-width="4"/>
</svg>'''
    st.markdown(
        f"""
<div style="border:1px solid {header_border};border-radius:12px;padding:14px 18px;background:{header_bg};margin-bottom:10px;display:flex;align-items:center;gap:16px;">
  <div style="flex-shrink:0;">{emblem_svg}</div>
  <div>
    <div style="font-size:40px;font-weight:700;color:{header_text};">Clinical Research Analysis (Beta)</div>
    <div style="font-size:20px;color:{header_subtext};">by Fadi Samaan</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    _show_pending_toast()  # Show any toast from previous action
    _render_sidebar_progress()

    # Main workflow sections - Data Preparation Phase
    st.markdown("## Data Preparation")

    with st.expander("Upload & Inspect", expanded=True):
        df = load_data_section()

    if df is None:
        st.stop()

    with st.expander("Data Filtering", expanded=False):
        df = data_filtering_section(df)

    with st.expander("Variable Transformations", expanded=False):
        df = variable_transformation_section(df)

    with st.expander("Graphing and Plots (Exploratory)", expanded=False):
        graphing_section()

    # Data Preparation Summary and Ready for Analysis
    with st.expander("Data Preparation Summary", expanded=True):
        analysis_ready = data_preparation_summary_section(df)

    # Only show analysis sections if user has clicked "Ready for Analysis"
    if not analysis_ready:
        st.info("Complete data preparation above and click 'Ready for Analysis' to proceed to the analysis sections.")
        st.stop()

    # Data Dictionary Section - Separate and easily accessible
    st.markdown("## Study Design")
    with st.expander("Data Dictionary & Variable Roles", expanded=True):
        data_dictionary_section(df)

    # Analysis Phase - Tab-based layout
    st.markdown("## Analysis")
    st.caption("Navigate between sections using the tabs below.")

    # Create main analysis tabs
    analysis_tabs = st.tabs([
        "📊 Tables",
        "🔬 Regression",
        "📈 Results Dashboard",
        "💾 Export"
    ])

    # Tab 1: Tables (Table 1, Table 2, Subgroup Analysis, Custom Summary)
    with analysis_tabs[0]:
        table_subtabs = st.tabs(["Table 1", "Table 2", "Subgroup Analysis", "Custom Summary"])

        with table_subtabs[0]:
            table1_section(df)

        with table_subtabs[1]:
            outcome_and_table2_section(df)

        with table_subtabs[2]:
            subgroup_analysis_section(df)

        with table_subtabs[3]:
            custom_table_builder_section(df)

    # Tab 2: Regression (Covariates, Regression)
    with analysis_tabs[1]:
        regression_subtabs = st.tabs(["Covariate Selection", "Multivariable Regression"])

        with regression_subtabs[0]:
            covariate_suggestion_section(df)
            st.session_state.progress["covariates"] = True if st.session_state.get("selected_covariates") else False

        with regression_subtabs[1]:
            regression_section(df)

    # Tab 3: Results Dashboard
    with analysis_tabs[2]:
        results_summary_dashboard(df)

    # Tab 4: Export
    with analysis_tabs[3]:
        report_and_export_section()


if __name__ == "__main__":
    main()
