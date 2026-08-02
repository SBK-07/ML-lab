# Experiment 3: Loan Amount Sanction Prediction using Linear & Regularized Regression

## Overview
This experiment focuses on predicting continuous loan sanction amounts (in USD) for applicants using Linear Regression and regularized regression models (Ridge, Lasso, and Elastic Net).

## Key Workflow & Methods
1. **Data Preprocessing & Imputation**:
   - Cleaned missing values across 11 features (mode imputation for categorical, median for skewed money features, mean for Credit Score).
   - Handled placeholder values (`-999` and `'?'`).
   - Encoded categorical variables via `LabelEncoder` and standardized numerical features using `StandardScaler`.
2. **Exploratory Data Analysis (EDA)**:
   - Generated a single-page consolidated **12-subplot EDA figure**, exported as `exp3_eda_12_subplots.eps` (600 DPI vector graphic).
3. **Model Building & Regularization Tuning**:
   - Baseline: Unregularized Linear Regression.
   - Tuned Ridge ($\alpha$), Lasso ($\alpha$), and Elastic Net ($\alpha$, $l1\_ratio$) using `GridSearchCV` with 5-fold cross-validation.
4. **Evaluation & Analysis**:
   - Compared MAE, MSE, RMSE, and $R^2$ metrics across models.
   - Analyzed coefficient shrinkage and bias-variance trade-off behavior.

## Key Results
- **Best Model**: Tuned Elastic Net ($\alpha=0.01, l1\_ratio=0.8$) achieved the best overall test performance ($R^2 = \mathbf{0.6554}$, $\text{RMSE} = 27800.90$).
- **Key Insight**: Regularization stabilized feature weight estimates (especially reducing heavy weight reliance on imputed Property Age) while maintaining strong predictive performance.

## Files Included
- `Regression_Analysis_Organized.ipynb`: Main Jupyter Notebook.
- `exp3_eda_12_subplots.eps`: 600 DPI vector EDA figure.
- `Experiment_3_Lab_Manual.pdf`: Official lab manual prompt.
