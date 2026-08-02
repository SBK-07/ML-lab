# Experiment 4: Binary Classification using Linear & Kernel-Based Models

## Overview
This experiment compares linear classification (Logistic Regression) against kernel-based classification (Support Vector Machines) for email spam detection on the Spambase dataset.

## Key Workflow & Methods
1. **Data Preprocessing**:
   - Deduplicated dataset (4,210 final rows; 60.12% ham, 39.88% spam).
   - Standardized features using `StandardScaler`.
2. **Exploratory Data Analysis (EDA)**:
   - Consolidated 12 EDA plots onto a single page figure grid, exported as `exp4_eda_12_subplots.eps` (600 DPI vector graphic).
3. **Model Building & Kernel Comparison**:
   - Evaluated Logistic Regression (L1-regularized via `liblinear`).
   - Compared SVM kernels: **Linear**, **Polynomial**, **RBF**, and **Sigmoid**.
   - Tuned $C$, penalty, solver (Logistic Regression) and $C$, $\gamma$, kernel (SVM) using 5-fold `GridSearchCV`.

## Key Results
- **Best Overall Model**: Tuned RBF Kernel SVM ($C=10, \gamma=0.01$) achieved **94.18% test accuracy** and **0.9268 F1-score**.
- **Kernel Comparison**: RBF kernel (**93.23%**) and Linear kernel (**93.59%**) significantly outperformed Polynomial (**76.60%**) and Sigmoid (**89.67%**) kernels.
- **Linear Baseline**: L1-regularized Logistic Regression achieved **93.71% test accuracy** with fast execution and high interpretability.

## Files Included
- `inary Classification using Linear and Kernel-Based Models.ipynb`: Main Jupyter Notebook.
- `exp4_eda_12_subplots.eps`: 600 DPI vector EDA figure.
- `Experiment_4.pdf`: Official lab manual prompt.
