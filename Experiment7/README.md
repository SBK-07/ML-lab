# Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Overview
This experiment systematically evaluates the impact of Principal Component Analysis (PCA) feature reduction on classifier performance across ten distinct model families (Linear, Margin-Based, Instance-Based, Decision Trees, and Ensembles) using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Key Workflow & Methods
1. **Data Preprocessing**:
   - Analyzed 569 samples across 30 continuous numerical features (0 missing values, 0 duplicate rows; 62.74% benign, 37.26% malignant).
   - Standardized features using `StandardScaler` fitted strictly on training data to prevent data leakage.
   - Applied 80/20 stratified train-test split (455 train, 114 test).
   - Applied PCA, identifying that **10 principal components** retain **95.27% of cumulative explained variance**.
2. **Exploratory Data Analysis (EDA)**:
   - Consolidated 12 EDA plots into a single figure grid formatted with Times New Roman 15 pt typography and bold labels, exported as `exp7_eda_12_subplots.eps` (600 DPI vector) and `exp7_eda_12_subplots.png`.
3. **Model Building & Evaluation**:
   - Evaluated 10 classification algorithms (SVM, Naive Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, and Stacking Classifier) both with raw standardized features and PCA-compressed features.
   - Performed 5-fold cross-validation hyperparameter tuning using `GridSearchCV`.

## Key Results
- **Top Performers (No-PCA)**: Linear SVM and Stacked Ensemble achieved **98.25% test accuracy**, **0.9861 F1-score**, and **>0.99 ROC-AUC**.
- **Impact of PCA**:
  - Distance & Margin-Based Models (SVM, KNN, Logistic Regression): Retained or mildly improved CV stability while reducing feature dimensionality by 66.7% (30 to 10 components).
  - Tree-Based Ensembles (Random Forest, AdaBoost, Gradient Boosting, XGBoost): Showed minor performance drops with PCA, confirming their inherent capability to perform feature selection without needing orthogonal transformation.

## Files Included
- `Experiment_7_PCA_Model_Evaluation.ipynb`: Primary Jupyter Notebook updated with `rules.md` compliant functions and plot formatting.
- `Experiment_7_Report.tex`: Full LaTeX report with 600 DPI EPS figure inclusions.
- `exp7_eda_12_subplots.eps` / `.png`: Consolidated 12-subplot EDA figure grid (600 DPI).
- `fig_pca_scree.eps`: Cumulative explained variance scree plot.
- `fig_performance_comparison.eps`: Model performance comparison chart across metrics.
- `fig_generalization_gap.eps`: Generalization gap comparison bar chart.
- `fig_roc_curves.eps`: ROC curve evaluation plot.
- `fig_confusion_matrix_stacking.eps`: Confusion matrix visualization.
