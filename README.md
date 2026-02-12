# Stats for Scrubs
# Clinical Research Analysis (Beta)

A comprehensive Streamlit-based application for epidemiological and biostatistical analyses on clinical datasets. Designed for researchers and clinicians to perform rigorous statistical analysis with an intuitive, guided workflow.

## Features

### Data Preparation
- **Data Upload**: Support for CSV and Excel files, plus built-in sample data for testing
- **Data Filtering**: Filter data by range, comparison operators, or categorical values
- **Variable Transformations**:
  - Log transform (natural, log10, log2)
  - Square root transform
  - Z-score standardization
  - Categorize continuous variables (quantiles, equal width, custom bins)
  - Dichotomize at threshold
  - Combine/recode categories
  - Winsorize outliers
  - Create interaction terms
- **Outlier Detection**: IQR and z-score methods
- **PHI Detection**: Automatic detection of potential identifiers

### Analysis
- **Data Dictionary**: Define variable roles (Exposure, Outcome, Covariate, etc.) and types
- **Table 1**: Baseline characteristics with standardized mean differences (SMD)
  - Support for both tableone engine and custom implementation
  - Mean (SD) or Median (IQR) for continuous variables
  - Missing data handling options
- **Table 2**: Outcome analysis with odds ratios and confidence intervals
- **Regression Analysis**:
  - Logistic regression (binary outcomes)
  - Linear regression (continuous outcomes)
  - Cox proportional hazards (time-to-event outcomes)
  - Crude vs. adjusted model comparison
  - Bootstrap confidence intervals
  - VIF for multicollinearity assessment
- **Subgroup Analysis**: Stratified regression with interaction testing
- **Survival Analysis**: Kaplan-Meier curves (requires lifelines)

### Visualization
- Forest plots for regression results
- Kaplan-Meier survival curves
- Histograms, boxplots, and scatter plots
- Exploratory data visualization

### Results Dashboard
- **All-in-one view**: See study parameters, Table 1 highlights, and regression results together
- **Key metrics display**: Exposure groups, outcomes, and covariate counts at a glance
- **SMD warnings**: Automatic highlighting of variables with potential imbalance
- **Integrated forest plot**: Visual representation of effect estimates
- **Quick export**: Download results directly from the dashboard

### Export & Reproducibility
- Excel export with formatted tables
- Word document reports (requires python-docx)
- JSON configuration export for reproducibility
- Analysis history tracking

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the repository:
```bash
cd /path/to/StatsAPP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Dependencies

### Required
- streamlit
- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- openpyxl

### Optional (for enhanced features)
- **tableone**: Enhanced Table 1 generation
- **lifelines**: Survival analysis and Kaplan-Meier curves
- **python-docx**: Word document report generation

Install optional dependencies:
```bash
pip install tableone lifelines python-docx
```

## Usage Guide

### Workflow Overview

The application follows a two-phase workflow with a modern tab-based interface:

#### Phase 1: Data Preparation (Expander-based)
1. **Upload & Inspect**: Load your dataset and review data quality metrics
2. **Data Filtering**: Apply filters to focus on specific subpopulations
3. **Variable Transformations**: Create derived variables as needed
4. **Graphing (Exploratory)**: Visualize your data before analysis
5. **Data Preparation Summary**: Review changes and click "Ready for Analysis"

#### Phase 2: Analysis (Tab-based Interface)

| Tab | Contents |
|-----|----------|
| 📋 **Data Dictionary** | Define variable roles and types |
| 📊 **Tables** | Table 1 (baseline), Table 2 (outcomes), Custom Summary |
| 🔬 **Regression** | Covariate selection, Multivariable models, Subgroup analysis |
| 📈 **Results Dashboard** | Summary of all results with forest plot and quick export |
| 💾 **Export** | Full export options, reproducibility tools, report generation |

### Quick Start

1. Click "Use sample data" to load demonstration data
2. Expand "Variable Transformations" to create any derived variables
3. Review the "Data Preparation Summary" and click "Ready for Analysis"
4. Navigate to the **Data Dictionary** tab and set one variable as "Exposure" and one as "Outcome"
5. Go to the **Tables** tab and generate Table 1 to see baseline characteristics
6. Use the **Regression** tab to select covariates and fit models
7. View all results at once in the **Results Dashboard** tab

### Variable Roles

| Role | Description |
|------|-------------|
| Exposure | Primary independent variable of interest |
| Outcome | Dependent variable(s) to analyze |
| Baseline covariate | Pre-exposure variables for adjustment |
| Post-exposure | Variables measured after exposure (not for adjustment) |
| Identifier | ID columns (excluded from analysis) |
| Ignore | Columns to exclude entirely |

### Variable Types

| Type | Description |
|------|-------------|
| Continuous | Numeric variables (age, weight, lab values) |
| Binary | Two-level categorical (yes/no, male/female) |
| Categorical | Multi-level categorical (race, treatment arm) |
| Ordinal | Ordered categorical (disease stage, education level) |
| Time-to-event component | Time or event indicator for survival analysis |

## Project Structure

```
StatsAPP/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── CLAUDE.md                 # Development guidelines
├── README.md                 # This file
└── utils/
    ├── __init__.py
    ├── statistics.py         # Statistical functions (SMD, VIF, model fitting)
    ├── tables.py             # Table 1 and Table 2 generation
    ├── export.py             # Excel, Word, JSON export utilities
    ├── transforms.py         # Data transformation functions
    ├── plots.py              # Visualization (forest plots, KM curves)
    └── history.py            # Undo/redo and analysis history
```

## Statistical Methods

### Standardized Mean Difference (SMD)
Used in Table 1 to assess covariate balance between exposure groups. SMD > 0.1 suggests meaningful imbalance.

### Regression Models
- **Logistic**: For binary outcomes, reports odds ratios (OR)
- **Linear**: For continuous outcomes, reports beta coefficients
- **Cox**: For time-to-event outcomes, reports hazard ratios (HR)

### Missing Data
- **Complete-case analysis**: Exclude observations with any missing values
- **Simple imputation**: Median for continuous, mode for categorical

### Confidence Intervals
- Default: Wald-based 95% CI
- Optional: Bootstrap 95% CI (slower but more robust)

## Tips for Clinical Research

1. **Always define your exposure and outcome before analysis**
2. **Check for multicollinearity** using VIF before fitting adjusted models
3. **Review SMD values** in Table 1 to identify imbalanced covariates
4. **Use baseline covariates only** - don't adjust for post-exposure variables
5. **Document your analysis** using the export features for reproducibility

## Limitations

- Not intended for regulatory submissions without validation
- Sample size calculations not included
- Advanced methods (propensity scores, multiple imputation) not yet implemented
- Time-to-event analysis requires the lifelines package

## Contributing

This is a beta version. Feedback and contributions are welcome.

## License

This project is provided as-is for research and educational purposes.

## Author

Fadi Samaan

---

*Clinical Research Analysis (Beta) v0.2.0*
