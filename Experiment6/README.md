
# Experiment 6: Bagging, Boosting, and Stacked Ensemble Models

## Overview
This experiment compares ensemble learning paradigms — Bootstrap Aggregation (Bagging Classifier), Sequential Boosting (AdaBoost and Gradient Boosting Classifiers), and Heterogeneous Stacked Ensembles (StackingClassifier) — on the Breast Cancer Wisconsin (Diagnostic) dataset to analyze performance, generalization gap, and bias–variance trade-offs.

## Key Workflow & Methods
1. **Data Preprocessing**:
   - Analyzed 569 samples across 30 continuous numerical features (0 missing values, 0 duplicate rows; 62.74% benign, 37.26% malignant).
   - Applied an 80/20 stratified train-test split (455 training rows, 114 test rows).
   - Scaled features with `StandardScaler` inside pipelines for distance and linear models (Logistic Regression and KNN base/meta learners).
2. **Exploratory Data Analysis (EDA)**:
   - Consolidated 12 distinct EDA plots onto a single page figure grid with Times New Roman 15 pt typography, exported as `exp6_eda_12_subplots.eps` (600 DPI vector) and `exp6_eda_12_subplots.png`.
3. **Model Building & Ensemble Tuning**:
   - Implemented reusable functions per `rules.md`: `plot_12_subplots_eda_v2`, `train_and_evaluate_classification_models`, and `compute_and_display_classification_metrics`.
   - Tuned Bagging hyperparameters (`n_estimators`, `max_samples`, `max_features`).
   - Tuned Boosting hyperparameters (`n_estimators`, `learning_rate`, `max_depth`).
   - Tuned Stacked Ensemble using heterogeneous base learners (Decision Tree, Logistic Regression, KNN) and a Logistic Regression meta-learner using 5-fold cross-validation.

## Key Results
- **Best Overall Model**: **Stacked Ensemble** achieved **98.25% test accuracy**, **100.0% recall**, **0.9863 F1-score**, and **0.9907 ROC-AUC** on held-out test data.
- **Boosting Models**: AdaBoost and Gradient Boosting both achieved **95.61% test accuracy**, **98.61% recall**, and **0.9660 F1-score**.
- **Bagging Classifier**: Achieved **93.86% test accuracy** and **0.9517 F1-score**.
- **Generalization Gap**: The Stacked Ensemble demonstrated the lowest generalization gap (0.43% difference between train and test accuracy) and highest CV stability (std dev 0.0078).

## Files Included
- `Bagging_Boosting_Stacked_Ensemble_Models_Organized.ipynb`: Primary Jupyter Notebook organized with reusable functions and Times New Roman plot styling.
