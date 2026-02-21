# Predicting HDL Cholesterol from NHANES Survey Variables

[![Competition](https://img.shields.io/badge/ASA-South%20Florida%20Data%20Challenge%202026-blue)](https://github.com/luminwin/ASASF)
[![Track](https://img.shields.io/badge/Track-Graduate%20Prediction-green)]()
[![Best Model](https://img.shields.io/badge/Best%20Model-CatBoost-orange)]()
[![CV RMSE](https://img.shields.io/badge/CV%20RMSE-4.5339-red)]()
[![R²](https://img.shields.io/badge/CV%20R²-0.7466-purple)]()

> **2026 ASA South Florida Student Data Challenge — Graduate Prediction Track**
>
> Can we predict HDL cholesterol from routinely collected survey data — without requiring laboratory lipid panels?

## Authors

| Name | Affiliation | Contact |
|------|------------|---------|
| **Daniel Regalado** | University of Miami | [dxr1491@miami.edu](mailto:dxr1491@miami.edu) |
| **Miguel Rocha** | University of Miami | |

---

## Table of Contents

- [Motivation](#motivation)
- [Competition Overview](#competition-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Motivation

Low HDL cholesterol is a well-established risk factor for cardiovascular disease, the leading cause of death worldwide. Direct measurement of HDL requires laboratory lipid panels that are costly and not universally available, particularly in resource-limited settings. This project explores whether routinely collected survey data — demographics, anthropometrics, dietary intake, and behavioral indicators — can serve as a reliable proxy for HDL estimation, enabling early screening and risk stratification at scale.

## Competition Overview

The [2026 ASA South Florida Student Data Challenge](https://github.com/luminwin/ASASF) is organized by the American Statistical Association's South Florida Chapter. Participants build predictive models using NHANES (National Health and Nutrition Examination Survey) data to forecast Direct HDL-Cholesterol (`LBDHDD_outcome`, mg/dL). Submissions are evaluated on **Root Mean Squared Error (RMSE)**, with the top 30% advancing to final judging based on report quality and analytical rigor.

## Dataset

| Split | Observations | Features | Missing Values |
|-------|-------------|----------|----------------|
| Train | 1,000 | 95 | None |
| Test  | 200 | 95 | None |

The 95 predictor variables are sourced from the 2024 NHANES cycle and span four domains:

- **Demographics** — age, gender, race/ethnicity, income-to-poverty ratio, marital status
- **Anthropometrics** — BMI, waist circumference
- **Dietary intake** — 24-hour recall data (macronutrients, micronutrients, food diversity)
- **Behavioral indicators** — alcohol consumption frequency and quantity

The target variable (`LBDHDD_outcome`) is a noise-adjusted version of Direct HDL-Cholesterol (mg/dL), approximately normally distributed (mean = 54.73, median = 54.16, skewness = 0.376).

## Repository Structure

```
.
├── README.md                          # This file
├── METHODS.md                         # Detailed methodology & technical decisions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
├── notebooks/
│   └── Submission_DanielRegalado_MiguelRocha.ipynb   # Full analysis notebook
├── reports/
│   └── DanielRegalado_MiguelRocha.pdf                # 3-page competition report
└── predictions/
    └── pred.csv                                       # Final test predictions (200 rows)
```

## Methodology

A full technical writeup is available in [`METHODS.md`](METHODS.md). The key stages are summarized below.

### 1. Exploratory Data Analysis

EDA revealed that waist circumference (*r* = -0.596) and BMI (*r* = -0.484) are the strongest linear predictors, confirming central adiposity as the primary driver of HDL variation. Gender showed a strong positive association (*r* = +0.523), with females exhibiting systematically higher HDL levels (mean 59.2 vs. 49.8 mg/dL). Dietary variables displayed weak individual correlations (|*r*| < 0.19), suggesting their effects operate through interactions rather than additive relationships.

### 2. Feature Engineering

A custom sklearn-compatible transformer (`FeatureEngineer`) generates 15 engineered features across four categories, all fitted exclusively on training data to prevent leakage:

| Category | Features | Rationale |
|----------|----------|-----------|
| Sex interactions | Waist x Sex, BMI x Sex, Age x Sex, Alcohol x Sex, Race x Sex, Income x Sex, Food Diversity x Sex, Fish x Sex | Capture gender-dependent effects from EDA |
| Body composition | Waist x BMI, Age x Waist | Model adiposity-age interactions |
| Polynomial terms | Waist², BMI², Age² | Capture nonlinear relationships |
| Log transforms | Skewed dietary variables | Normalize right-skewed distributions |

Permutation importance analysis identified and removed 37 features (34 raw + 3 engineered) with zero or negative contribution, reducing dimensionality from 107 to 73 features.

### 3. Model Selection

Eight model configurations were compared using 5-fold cross-validation. Hyperparameters for tree-based models were optimized via Optuna Bayesian optimization.

| Model | CV RMSE | CV MAE | CV R² |
|-------|---------|--------|-------|
| **CatBoost** | **4.5339** | **3.6680** | **0.7466** |
| Stacking (Cat+XGB) | 4.5394 | 3.6698 | 0.7460 |
| Stacking (Cat+XGB+LGB) | 4.5510 | 3.6774 | 0.7446 |
| XGBoost | 4.5869 | 3.7059 | 0.7406 |
| LightGBM | 4.6255 | 3.7123 | 0.7362 |
| Elastic Net | 5.4302 | 4.2123 | 0.6356 |
| NN with Dropout | 5.7620 | — | — |
| NN Standard | 6.8587 | — | — |

### 4. Interpretability

SHAP analysis of the final CatBoost model reveals that engineered interaction features account for 12 of the top 20 predictors by mean |SHAP| value, validating the EDA-guided feature engineering strategy.

## Results

The final CatBoost model achieves **CV RMSE = 4.5339** and **R² = 0.7466** on 5-fold cross-validated out-of-fold predictions. Residual diagnostics confirm well-calibrated predictions: errors are centered near zero (mean = +0.054), approximately normal (skewness = 0.171), and only 3.3% of observations exceed the 2-sigma threshold. Prediction accuracy is highest in the normal HDL range (50–60 mg/dL) and degrades at extremes (>70 mg/dL), consistent with unobserved genetic and medication factors absent from the feature space.

**Test prediction summary:** mean = 54.41 mg/dL, std = 7.60, range = [40.15, 75.31] mg/dL.

## Reproducibility

### Requirements

```bash
# Clone the repository
git clone https://github.com/<your-username>/hdl-cholesterol-prediction.git
cd hdl-cholesterol-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

Open and execute the Jupyter notebook:

```bash
jupyter notebook notebooks/Submission_DanielRegalado_MiguelRocha.ipynb
```

> **Note:** The training and test datasets are provided by the competition organizers and are not included in this repository. See the [competition page](https://github.com/luminwin/ASASF) for data access.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  <i>University of Miami — MAS 651 — Spring 2026</i>
</p>
