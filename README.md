# Credit Card Default Prediction

## Overview
This project builds a machine learning model to predict credit card default using demographic data and payment history from 30,000 customers. The goal is to help banks identify high-risk customers early and intervene before default occurs.

## Dataset
- **Source**: [UCI Credit Card Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Size**: 30,000 customers, 23 original features + 3 engineered features
- **Target**: Whether a customer will default next month (1 = default, 0 = no default)
- **Default rate**: 22.1%

## Project Structure
```
├── notebook.ipynb       # Main analysis notebook
└── README.md
```

## Workflow
1. **Exploratory Data Analysis (EDA)** — Understanding customer demographics and payment behavior
2. **Data Cleaning** — Handling undefined categorical values and negative bill amounts
3. **Feature Engineering** — Creating 3 new features to improve model performance
4. **Modeling** — Comparing Logistic Regression and Random Forest
5. **Hyperparameter Tuning** — GridSearchCV optimizing for recall
6. **Business Recommendations** — Threshold selection based on intervention type

## Key Findings
- **Most important predictor**: PAY_0 (most recent repayment status)
- **Gender and marital status** show minimal predictive power
- **Credit utilization ratio** (BILL_UTIL) is the second most important feature

## Model Performance

| Model | Threshold | Recall | Precision | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.5 | 0.605 | 0.374 | 0.710 |
| Random Forest | 0.4 | 0.445 | 0.575 | 0.756 |
| **Optimized RF (final)** | **0.4** | **0.697** | **0.401** | **0.778** |

## Business Recommendations
Two-tier early warning system based on intervention severity:

| Scenario | Model | Threshold | Defaults Captured | Customers Flagged |
|---|---|---|---|---|
| Reminder email | Optimized RF | 0.3 | 86.7% (1,150/1,327) | 2,572 |
| Credit limit reduction | Optimized RF | 0.5 | 58.0% (769/1,327) | 741 |

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, GridSearchCV)
- Matplotlib, Seaborn

## Kaggle Notebook
[View on Kaggle](https://www.kaggle.com/code/panyouzhen/notebook1046aababe)
