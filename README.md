# Credit Default Prediction

A machine learning project predicting the probability of a borrower defaulting on a loan (90+ day delinquency within 2 years).

---

## Overview

Banks lose money when borrowers fail to repay loans. This project builds a classification model that estimates default risk for each applicant, allowing lenders to make informed decisions before approving credit.

**Task:** Binary classification — will this client default within 2 years?  
**Metric:** ROC-AUC (accuracy is misleading due to class imbalance)  
**Best result:** ROC-AUC = 0.627 with CatBoost + probability calibration

---

## Dataset

15,000 clients with 11 features including credit utilization, income, debt ratio, and delinquency history. Class distribution: 83% non-default / 17% default.

Source: Give Me Some Credit (Kaggle)

---

## What Was Done

**Data Cleaning**
- Filled missing values in `MonthlyIncome` (20%) and `NumberOfDependents` (2.5%) with median
- Removed impossible values: `age = 0`, credit utilization above 1.0, error code `96` in delinquency columns
- Clipped outliers in `monthly_debt` and `income_per_person` at the 99th percentile

**Feature Engineering**
- `total_past_due` — sum of all delinquency columns
- `monthly_debt` — actual debt amount (income × debt ratio)
- `income_per_person` — disposable income per family member
- `has_real_estate` — binary flag for property ownership
- `credit_load` — number of open credit lines relative to age

`total_past_due` ranked top 4 in SHAP importance, validating the feature engineering effort.

**Modeling — Iteration 1**

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.6014 |
| XGBoost default | 0.5975 |
| XGBoost tuned | 0.6138 |
| XGBoost + early stopping | 0.6150 |

**Modeling — Iteration 2**

| Model | ROC-AUC | Notes |
|---|---|---|
| LightGBM + early stopping | 0.6026 | Unstable on small datasets |
| CatBoost default | 0.6270 | Best out-of-the-box performance |
| CatBoost + Optuna (30 trials) | 0.6231 | Undertuned — needs 100+ trials |
| CatBoost + Platt Scaling | **0.6270** | Best calibrated model |

**Handling Class Imbalance**  
Tested `class_weight='balanced'`, threshold tuning (0.17), and SMOTE oversampling. All methods improved recall for the minority class without significant ROC-AUC gain — indicating the main bottleneck was model capacity, not imbalance.

`scale_pos_weight = 4.85` used in all gradient boosting models (ratio of negative to positive class).

**Cross-Validation**  
Used `StratifiedKFold(n_splits=5)` to ensure consistent class distribution across all folds.

Results confirmed CatBoost stability: std = 0.0078 vs LightGBM std = 0.012.

**Hyperparameter Tuning**

*Iteration 1 — RandomizedSearchCV:*  
20 combinations. Best: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.6`.

*Iteration 2 — Optuna (Bayesian optimization):*  
30 trials with `direction='maximize'` on ROC-AUC.  
Best found: `depth=7`, `learning_rate=0.015`, `subsample=0.73`, `l2_leaf_reg=1.02`.  
Most important hyperparameter: `learning_rate` (importance = 0.72).

> Note: 30 trials is insufficient for stable convergence. Production tuning requires 100–200 trials minimum.

**Early Stopping**  
- XGBoost: trained with 500 trees, stopped at tree #38
- CatBoost: stopped at tree #31
- Reduced train/test AUC gap from 0.073 to acceptable range

**Model Interpretability (SHAP)**  
Top features by mean absolute SHAP value:

| Feature | SHAP Importance | Direction |
|---|---|---|
| DebtRatio | 0.20 | High debt → higher risk |
| RevolvingUtilizationOfUnsecuredLines | 0.14 | High utilization → higher risk |
| age | 0.10 | Younger → higher risk |
| total_past_due | 0.07 | More delinquencies → higher risk |
| credit_load | 0.05 | More loans per age → higher risk |
| has_real_estate | ~0.00 | Not predictive |

SHAP allows explaining individual predictions — required for regulatory compliance in real banking environments (Basel III, Central Bank requirements).

**Probability Calibration**

Raw model probabilities were systematically overestimated (curve below diagonal on calibration plot).

Two methods tested:
- **Isotonic Regression** — overfitted on small validation set (unstable curve)
- **Platt Scaling** (logistic regression over raw probabilities) — stable, closer to diagonal

Calibration does not improve ROC-AUC (ranking order unchanged) but makes the probability values accurate in absolute terms — critical for risk provisioning in banking.

---

## Key Findings

- Borrowers spending more than 50% of income on debt are significantly more likely to default
- Credit utilization above 80% is a strong default signal
- Younger borrowers (under 35) show higher default rates
- Early stopping is critical — XGBoost peaked at tree #38, everything after added noise
- CatBoost outperforms XGBoost and LightGBM out of the box on this dataset
- `has_real_estate` showed near-zero SHAP importance — property ownership alone is not predictive
- Platt Scaling is preferred over Isotonic Regression when calibration data is limited

---

## Project Structure

```
credit-scoring/
    data/
        cs-training.csv        # place dataset here
    notebooks/
        01_eda.ipynb
    README.md
    .gitignore
```

---

## Stack

Python 3.13, pandas, numpy, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, imbalanced-learn, SHAP, matplotlib, seaborn

---

## How to Run

```bash
git clone https://github.com/your-username/credit-scoring.git
cd credit-scoring
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna imbalanced-learn shap matplotlib seaborn
jupyter notebook notebooks/01_credit_scoring.ipynb
```

Place `cs-training.csv` in the `data/` folder before running.
