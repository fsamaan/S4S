# User Guide

A step-by-step guide to using the Clinical Research Analysis application.

---

## Getting Started

1. Launch the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to `http://localhost:8501`

---

## Phase 1: Data Preparation

Complete these steps before proceeding to analysis.

### Step 1: Upload Your Data

- Click **Use sample data** to try the app with demonstration data, or
- Upload your own CSV or Excel file using the file uploader

Once loaded, review:
- Row and column counts
- Data types
- Percent missing for each variable
- Data readiness checklist (duplicates, constant columns, high missingness)

### Step 2: Filter Data (Optional)

Use the **Data Filtering** section to subset your data:
- Filter by numeric range (e.g., age between 18 and 65)
- Filter by comparison (e.g., BMI > 25)
- Filter by category (e.g., gender = "Female")

Click **Apply Filter** after each filter. Multiple filters can be combined.

### Step 3: Transform Variables (Optional)

Use the **Variable Transformations** section to create derived variables:

| Transformation | Use Case |
|----------------|----------|
| Log transform | Normalize skewed distributions |
| Square root | Stabilize variance |
| Z-score | Standardize for comparison |
| Categorize | Convert continuous to categorical (e.g., age groups) |
| Dichotomize | Create binary variables from continuous |
| Combine categories | Merge rare categories together |
| Winsorize | Handle outliers by capping extreme values |
| Interaction term | Create variable interactions for regression |

### Step 4: Review and Proceed

In the **Data Preparation Summary** section:
- Review the number of new variables created
- Check any active filters
- Click **Ready for Analysis** to proceed

---

## Phase 2: Analysis

After clicking "Ready for Analysis", five tabs become available.

### Tab 1: Data Dictionary

Define how each variable should be used in your analysis.

**Variable Roles:**
| Role | Description |
|------|-------------|
| Exposure | Your primary independent variable of interest |
| Outcome | The dependent variable(s) you want to analyze |
| Baseline covariate | Pre-exposure variables to adjust for |
| Post-exposure | Variables measured after exposure (not for adjustment) |
| Identifier | ID columns (excluded from analysis) |
| Ignore | Columns to exclude entirely |

**Variable Types:**
| Type | Description |
|------|-------------|
| Continuous | Numeric (age, weight, lab values) |
| Binary | Two levels (yes/no, male/female) |
| Categorical | Multiple levels (race, treatment arm) |
| Ordinal | Ordered categories (disease stage) |
| Time-to-event | For survival analysis |

**Important:** Set at least one Exposure and one Outcome before generating tables.

### Tab 2: Tables

#### Table 1 (Baseline Characteristics)
- Displays summary statistics stratified by exposure group
- Shows counts and percentages for categorical variables
- Shows mean (SD) or median (IQR) for continuous variables
- Calculates Standardized Mean Difference (SMD) to assess balance
- SMD > 0.1 suggests potential imbalance

#### Table 2 (Outcome Analysis)
- Cross-tabulates exposure and outcome
- Calculates odds ratios with 95% confidence intervals
- Provides chi-square test p-values

#### Subgroup Analysis
- Perform stratified analyses by categorical variables
- Test for interaction effects
- Identify effect modification

#### Custom Summary
- Build custom summary tables with selected variables

### Tab 3: Regression

#### Covariate Selection
- Review suggested covariates based on your data dictionary
- Select which variables to include in adjusted models
- Check Variance Inflation Factor (VIF) for multicollinearity

#### Multivariable Regression
- Fits appropriate model based on outcome type:
  - Binary outcome → Logistic regression (Odds Ratios)
  - Continuous outcome → Linear regression (Beta coefficients)
  - Time-to-event → Cox regression (Hazard Ratios)
- Compare crude vs. adjusted estimates
- Option for bootstrap confidence intervals

### Tab 4: Results Dashboard

View all results in one place:
- Study parameters summary
- Key metrics (exposure groups, outcomes, covariates)
- Table 1 highlights with SMD warnings
- Regression results with forest plot
- Quick export options

### Tab 5: Export

Export your results in multiple formats:
- **Excel**: Formatted workbook with all tables
- **Word**: Professional report document
- **JSON**: Configuration file for reproducibility

---

## Quick Reference

### Recommended Workflow

1. Upload data
2. Apply filters (if needed)
3. Create derived variables (if needed)
4. Click "Ready for Analysis"
5. Define exposure and outcome in Data Dictionary
6. Generate Table 1
7. Select covariates
8. Run regression
9. Review Results Dashboard
10. Export results

### Tips

- Always define your exposure and outcome before running analyses
- Check SMD values in Table 1 to identify imbalanced covariates
- Use VIF to detect multicollinearity before fitting adjusted models
- Only adjust for baseline (pre-exposure) covariates
- Use the Results Dashboard for a comprehensive overview

### Theme Toggle

Switch between light and dark mode using the toggle in the sidebar.

---

## Support

For issues or feedback, visit the project repository.

---

*Clinical Research Analysis (Beta) v0.2.0*
