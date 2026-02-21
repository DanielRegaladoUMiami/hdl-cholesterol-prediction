# Technical Methodology

> Detailed description of all analytical decisions, feature engineering, model selection, and evaluation strategies used in this project.

## Table of Contents

- [1. Data Overview](#1-data-overview)
- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
- [3. Feature Engineering Pipeline](#3-feature-engineering-pipeline)
- [4. Feature Selection](#4-feature-selection)
- [5. Modeling Strategy](#5-modeling-strategy)
- [6. Hyperparameter Optimization](#6-hyperparameter-optimization)
- [7. Ensemble Methods](#7-ensemble-methods)
- [8. Neural Network Experiments](#8-neural-network-experiments)
- [9. Model Evaluation & Residual Diagnostics](#9-model-evaluation--residual-diagnostics)
- [10. Interpretability](#10-interpretability)

---

## 1. Data Overview

The dataset consists of 1,200 observations (1,000 train / 200 test) with 95 NHANES predictor variables and one continuous target: Direct HDL-Cholesterol (`LBDHDD_outcome`, mg/dL). No missing values are present in either split. The target is approximately normally distributed (mean = 54.73, median = 54.16, skewness = 0.376), eliminating the need for target transformation.

## 2. Exploratory Data Analysis

EDA was structured around three objectives: identifying the strongest marginal predictors, detecting subgroup heterogeneity, and assessing whether dietary variables contribute independently or through interactions.

**Key findings:**

- **Waist circumference** (*r* = -0.596) and **BMI** (*r* = -0.484) emerged as the strongest linear predictors, confirming that central adiposity is the dominant driver of HDL variation in this cohort.
- **Gender** (*r* = +0.523) showed pronounced group-level separation: females average 59.2 mg/dL vs. 49.8 mg/dL for males, a ~9.4 mg/dL gap consistent with established endocrine mechanisms.
- **Dietary variables** exhibited uniformly weak marginal correlations (|*r*| < 0.19), suggesting their predictive value is conditional on demographic covariates — motivating interaction-based feature engineering.
- **Alcohol consumption** displayed a nonlinear, sex-dependent relationship with HDL, with moderate intake associated with higher HDL predominantly in females.

These patterns directly informed the feature engineering strategy.

## 3. Feature Engineering Pipeline

A custom sklearn-compatible transformer (`FeatureEngineer`) was implemented with a leakage-safe design: all statistical thresholds and parameters are computed exclusively during `fit()` on training data.

### 3.1 Sex Interaction Terms

Given the strong gender-dependent effects identified in EDA, eight interaction terms were created by multiplying key predictors with the binary sex indicator:

`Waist x Sex`, `BMI x Sex`, `Age x Sex`, `Alcohol x Sex`, `Race x Sex`, `Income x Sex`, `Food Diversity x Sex`, `Fish x Sex`

### 3.2 Body Composition Interactions

Two multiplicative interactions capture the joint effect of adiposity measures:

`Waist x BMI`, `Age x Waist`

### 3.3 Polynomial Terms

Second-degree polynomial features for the three strongest continuous predictors:

`Waist²`, `BMI²`, `Age²`

### 3.4 Log Transforms

Logarithmic transformations applied to highly right-skewed dietary intake variables to compress their dynamic range and improve model sensitivity to variation in the lower quantiles.

## 4. Feature Selection

Permutation importance (computed on cross-validated out-of-fold predictions) identified 37 features with zero or negative contribution:

| Category | Count | Examples |
|----------|-------|---------|
| Dietary details | 27 | DR1EXMER, DR1_300, DR1TS060, DR1TBCAR, DR1TCHL |
| Survey / behavioral | 7 | WTDR2D, DRQSPREP, DBQ095Z, DR1DAY, DRQSDIET |
| Engineered (harmful) | 3 | High_Waist, Is_Obese, Waist_to_BMI |

Removing these features reduced the input space from 107 to **73 features**, improving both generalization performance and training efficiency.

## 5. Modeling Strategy

For the final submission, all 1,000 training observations were used for model fitting — no held-out validation split — to maximize information available to each model. Performance estimates were obtained exclusively through **5-fold cross-validation with out-of-fold predictions**, providing unbiased error estimates while allowing all data for final training.

### Models Evaluated

1. **Elastic Net** — L1+L2 regularized linear regression (baseline)
2. **CatBoost** — Gradient boosting with ordered boosting and symmetric trees
3. **XGBoost** — Gradient boosting with histogram-based splits
4. **LightGBM** — Gradient boosting with leaf-wise growth
5. **Stacking (Cat+XGB)** — Two-model stack with Ridge meta-learner
6. **Stacking (Cat+XGB+LGB)** — Three-model stack with Ridge meta-learner
7. **Neural Network (Standard MLP)** — Fully connected network
8. **Neural Network (with Dropout)** — MLP with dropout regularization

## 6. Hyperparameter Optimization

Tree-based models were optimized using **Optuna Bayesian optimization** with 5-fold CV as the objective. Elastic Net used `GridSearchCV`. Neural networks used Optuna with early stopping on an internal 85/15 holdout.

### Best Hyperparameters

| Model | Configuration | CV RMSE |
|-------|--------------|---------|
| CatBoost | iterations=1596, depth=6, lr=0.0098, l2_leaf_reg=0.09 | 4.5339 |
| XGBoost | n_estimators=1969, max_depth=4, lr=0.0040, colsample=0.52 | 4.5869 |
| LightGBM | n_estimators=439, max_depth=6, lr=0.0320, num_leaves=24 | 4.6255 |
| NN Dropout | 2 layers (64, 256), dropout=0.4, lr=0.0028, batch=32 | 5.4529 |

## 7. Ensemble Methods

Two stacking configurations were evaluated, both using **Ridge regression** as the meta-learner trained on out-of-fold predictions from the base models:

- **Stacking_2**: CatBoost + XGBoost (CV RMSE = 4.5394)
- **Stacking_3**: CatBoost + XGBoost + LightGBM (CV RMSE = 4.5510)

Neither ensemble improved upon standalone CatBoost (4.5339), indicating that the three gradient boosting variants extract overlapping rather than complementary predictive patterns from the engineered feature space. CatBoost's ordered boosting mechanism and internal regularization proved sufficient for this dataset.

## 8. Neural Network Experiments

Two MLP architectures were tested using TensorFlow/Keras, with Optuna optimizing layer count, unit sizes, learning rate, and batch size:

- **Standard MLP**: CV RMSE = 6.8587 — high variance due to the moderate sample-to-feature ratio (1,000 observations, 73 features)
- **MLP with Dropout** (p=0.4): CV RMSE = 5.7620 — dropout regularization substantially improved generalization

Neural networks underperformed tree-based methods, consistent with established findings that gradient boosting dominates on structured tabular datasets of this size.

## 9. Model Evaluation & Residual Diagnostics

Residual diagnostics for the final CatBoost model were computed on cross-validated out-of-fold predictions across all 1,000 training observations:

| Statistic | Value |
|-----------|-------|
| Mean Residual | +0.054 |
| Std Deviation | 4.534 |
| Skewness | 0.171 |
| Outliers (\|r\| > 2σ) | 33/1,000 (3.3%) |
| CV RMSE | 4.5339 |
| CV MAE | 3.6680 |
| CV R² | 0.7466 |

Errors are centered near zero, approximately normal, and show no systematic pattern against predicted values. Prediction accuracy is highest in the normal range (HDL 50–60 mg/dL) and deteriorates at extremes (HDL > 70 mg/dL), attributable to unobserved factors (genetics, medications, exercise) absent from the feature space.

## 10. Interpretability

### SHAP Analysis

Global SHAP importance from the CatBoost model confirms EDA findings and validates feature engineering:

| Rank | Feature | Mean \|SHAP\| | Cumulative % |
|------|---------|---------------|-------------|
| 1 | Gender (RIAGENDR) | 1.516 | 12.8% |
| 2 | Waist Circumference | 1.121 | 22.3% |
| 3 | Waist² | 1.039 | 31.1% |
| 4 | Waist x BMI | 1.009 | 39.6% |
| 5 | Age x Sex | 0.745 | 45.9% |
| 6 | Alcohol x Sex | 0.722 | 52.0% |
| 7 | Alcohol (DR1TALCO) | 0.718 | 58.1% |
| 8 | Income x Sex | 0.714 | 64.1% |
| 9 | Alc Freq x Amount | 0.557 | 68.8% |
| 10 | BMI (BMXBMI) | 0.515 | 73.2% |

Engineered features occupy **12 of the top 20 SHAP positions**, demonstrating that clinically-motivated interaction terms successfully encode conditional effects that tree models exploit for improved prediction.

### LIME Analysis

Local LIME explanations are consistent with SHAP: accurate predictions arise when feature contributions balance around typical training patterns, while large errors occur when multiple strong contributors reinforce each other near distribution extremes.

---

*For the complete analysis with code and visualizations, see the [Jupyter notebook](notebooks/Submission_DanielRegalado_MiguelRocha.ipynb).*
