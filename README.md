# Employee Attrition Prediction & Workforce Risk Analysis

## Overview

This project develops a supervised machine learning solution to predict whether an employee is likely to leave an organization.

The analysis focuses on employee demographics, compensation, engagement, workplace characteristics, and tenure-related factors. The project also examines model generalization, overfitting, class imbalance, feature importance, and the trade-off between predictive performance and interpretability.

## Problem Statement

Employee attrition can affect productivity, operational continuity, and recruitment costs.

The objective is to predict employee attrition (`Yes` / `No`) and provide a data-driven approach that can support proactive workforce and retention decisions.

Because employees who leave represent the minority class, the analysis prioritizes:

1. Recall for the attrition (`Yes`) class
2. F1-score
3. ROC-AUC

Accuracy is reported but is not used as the primary model-selection criterion.

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance dataset**.

- 1,470 employees
- 39 original columns
- 237 employees left the organization
- 1,233 employees stayed
- Attrition rate: approximately 16.1%

The dataset contains demographic, organizational, compensation, performance, and workplace-related attributes.

## Data Preparation & Feature Engineering

The preprocessing workflow included:

- Missing-value and duplicate checks
- Column-by-column data quality review
- Removal of zero-variance columns
- Removal of identifier and noise columns
- Detection and removal of a target-leakage column
- Categorical feature encoding
- One-hot encoding of nominal variables
- Standardization using `StandardScaler`
- Stratified train-test splitting
- SMOTE applied only to the training data

Five engineered features were created:

- `TenureRatio`
- `IncomePerJobLevel`
- `PromotionGap`
- `IsFrequentTraveler`
- `AvgSatisfaction`

These features were designed to capture interpretable relationships involving tenure, compensation, promotion history, travel, and employee satisfaction.

## Exploratory Data Analysis

The analysis identified several patterns associated with employee attrition:

- Employees who leave tend to be younger, lower-income, and shorter-tenured.
- `OverTime` was the strongest categorical signal of attrition.
- Frequent travel and single marital status were associated with higher attrition.
- Sales-related departments and certain technical roles showed elevated attrition rates.
- `TenureRatio` and `PromotionGap` showed visible separation between attrition classes.

Correlation analysis also highlighted relationships among tenure, income, and age variables.

## Machine Learning Models

Multiple supervised learning algorithms were evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting

Logistic Regression was used as the interpretable baseline, while the other algorithms were evaluated as alternative non-linear models.

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the non-linear models.

The tuning process explored parameters such as:

- Decision Tree depth and minimum sample settings
- KNN number of neighbors and weighting
- Random Forest depth, number of estimators, and split parameters
- Gradient Boosting learning rate, depth, and number of estimators

The analysis also examined training-versus-validation performance to identify potential overfitting.

## Model Evaluation

On the untouched test set, the reported results were:

| Model | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.702 | 0.516 | 0.809 |
| Decision Tree (Tuned) | 0.511 | 0.453 | 0.651 |
| Gradient Boosting (Tuned) | 0.319 | 0.375 | 0.797 |
| Random Forest (Tuned) | 0.234 | 0.319 | 0.811 |
| KNN (Tuned) | 0.468 | 0.312 | 0.640 |

Logistic Regression achieved the highest Recall and F1-score on the real test set while maintaining an ROC-AUC comparable to the ensemble models.

## Generalization & Overfitting

The project explicitly evaluates the difference between training and test performance.

The analysis found that Logistic Regression had the smallest train-test F1 gap, while Random Forest and KNN showed very large gaps associated with near-perfect training performance but substantially weaker test performance.

This demonstrates that strong cross-validation performance on SMOTE-resampled data does not necessarily guarantee strong generalization to the original imbalanced population.

## Final Model Recommendation

Based on the predefined evaluation priorities of:

**Recall → F1-score → ROC-AUC**

Logistic Regression was selected as the recommended final model.

It achieved:

- Recall: **0.702**
- F1-score: **0.516**
- ROC-AUC: **0.809**

The model also provides an interpretable coefficient-based explanation of employee attrition risk.

## Business Interpretation

The analysis suggests that factors such as overtime, frequent travel, tenure, and employee satisfaction can provide useful signals for identifying potential attrition risk.

A practical HR application could use model predictions as a **decision-support tool** to help identify employees who may benefit from retention discussions or further engagement analysis.

The model should complement rather than replace managerial judgement.

## Limitations & Future Work

Important limitations include:

- Class imbalance
- Potential overfitting in more complex models
- Differences between SMOTE-based cross-validation and the real test population
- Dataset-specific patterns that may not generalize to other organizations

Potential future improvements include:

- Threshold tuning
- Class-weighted models as an alternative to SMOTE
- SHAP-based individual-level explanations
- Validation on real-world organizational attrition data

## Repository Contents

| File | Description |
|---|---|
| `employee_attrition_prediction.ipynb` | Complete machine learning analysis and modelling notebook |
| `employee_attrition_prediction_report.docx` | Final project report |
| `README.md` | Project documentation |

## Tools & Technologies

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- SMOTE

## Reproducibility

The project uses a fixed random state (`RANDOM_STATE = 42`) for the train-test split, SMOTE, and model experiments.

The documented pipeline follows:

`StandardScaler → SMOTE (training data only) → Model`

## Academic Project

This project was completed as part of **Advanced Apex Project-2, Problem Statement P1: Employee Attrition Prediction & Workforce Risk Analysis**.
