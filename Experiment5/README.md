# Experiment 5: Binary Classification using Decision Trees and Random Forests

## Overview
This experiment compares non-linear decision boundary classification (Decision Tree Classifier) against ensemble bagging classification (Random Forest Classifier) for predicting breast cancer tumor malignancy on the Breast Cancer Wisconsin (Diagnostic) dataset.

## Key Workflow & Methods
1. **Data Preprocessing**:
   - Analyzed 569 samples across 30 continuous numerical features (0 missing values, 0 duplicate rows; 62.74% benign, 37.26% malignant).
   - Standardized features using `StandardScaler` for scale-invariant distance and importance comparisons.
   - Applied an 80/20 stratified train-test split (455 training rows, 114 test rows).
2. **Exploratory Data Analysis (EDA)**:
   - Consolidated 12 distinct EDA plots onto a single page figure grid, exported as `exp5_eda_12_subplots.eps` (600 DPI vector) and `exp5_eda_12_subplots.webp`.
3. **Model Building & Ensemble Tuning**:
   - Implemented reusable modular functions for EDA (`plot_12_subplots_eda`), model training (`train_and_evaluate_classification_model`), and metric evaluation (`compute_classification_metrics`).
   - Tuned Decision Tree hyperparameters (`criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
   - Tuned Random Forest ensemble parameters (`n_estimators`, `max_depth`, `max_features`, `bootstrap`) using 5-fold Stratified `GridSearchCV`.

## Key Results
- **Best Overall Model**: Tuned Random Forest Classifier (`n_estimators=100`, `max_depth=5`, `max_features='sqrt'`) achieved **96.49% test accuracy**, **0.9524 F1-score**, and **0.9950 ROC-AUC**.
- **Decision Tree Baseline**: Tuned Decision Tree Classifier (`criterion='gini'`, `max_depth=5`) achieved **95.61% test accuracy** and **0.9412 F1-score**.
- **Ensemble Advantage**: Random Forest reduced decision tree variance, improved false positive specificity (reducing false alarms from 3 to 2), and delivered near-perfect probability calibration (AUC 0.9950 vs 0.9554).
- **Dominant Features**: Gini feature importance identified `worst concave points`, `worst perimeter`, `worst radius`, and `mean concave points` as the primary predictors of tumor malignancy.

## Files Included
- `Breast_Cancer_Classification_using_Decision_Trees_and_Random_Forests.ipynb`: Primary Jupyter Notebook.
- `Experiment5.ipynb`: Executable notebook alias.
- `main.tex`: Full LaTeX report structured per `Manual (1).tex` and `rules.md`.
- `exp5_eda_12_subplots.eps` / `.webp`: Consolidated 12-subplot EDA figure grid.
- `exp5_confusion_matrices.eps` / `.webp`: Side-by-side Decision Tree vs Random Forest confusion matrices (compact 0.66 MB EPS).
- `exp5_roc_curves.eps` / `.webp`: Combined ROC curves comparison.
- `exp5_metrics_comparison.eps` / `.webp`: Test performance metrics bar chart.
- `exp5_feature_importances.eps` / `.webp`: Top 10 Gini feature importances.
- `exp5_decision_tree_structure.eps` / `.webp`: Tuned decision tree hierarchy diagram.
