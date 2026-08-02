# Experiment 2: Email Spam/Ham Classification using Naïve Bayes & KNN

## Overview
This experiment builds and evaluates machine learning classification models to distinguish spam emails from normal (ham) emails using the Spambase dataset. We compare Naïve Bayes variants (Gaussian, Multinomial, Bernoulli) against K-Nearest Neighbours (KNN) with hyperparameter tuning and search algorithm optimization.

## Key Workflow & Methods
1. **Data Cleaning & Preprocessing**:
   - Removed 391 duplicate rows from 4,601 initial records (final dataset: 4,210 rows; 60% ham, 40% spam).
   - Applied `StandardScaler` for distance-based KNN distance calculations.
2. **Exploratory Data Analysis (EDA)**:
   - Generated a consolidated **12-subplot EDA summary figure** on a single page, exported as `exp2_eda_12_subplots.eps` (600 DPI vector graphic).
3. **Model Building & Tuning**:
   - Evaluated Gaussian, Multinomial, and Bernoulli Naïve Bayes models.
   - Tuned KNN ($k=9$, distance weighting, Manhattan distance) using `GridSearchCV` and `RandomizedSearchCV`.
   - Compared spatial indexing methods: **KDTree** vs. **BallTree**.
   - Evaluated 5-fold cross-validation performance.

## Key Results
- **Best Overall Model**: Tuned KNN ($k=9$, Manhattan distance) achieved **92.76% accuracy** and **0.9039 F1-score**.
- **Best Naïve Bayes Variant**: Bernoulli NB achieved **89.43% accuracy** and **0.8658 F1-score**.
- **Search Efficiency**: BallTree was slightly faster than KDTree during training and prediction while maintaining identical accuracy.

## Files Included
- `Email_Spam_or_Ham_Classification_using_Naïve_Bayes_and_KNN_organised_version.ipynb`: Main Jupyter Notebook.
- `exp2_eda_12_subplots.eps`: 600 DPI vector EDA figure.
- `Experiment_2.pdf`: Official lab manual prompt.
