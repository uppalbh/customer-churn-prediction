# Customer Churn Prediction & Revenue-at-Risk Analysis

## Overview

Customer churn is one of the most significant challenges faced by subscription-based businesses. Acquiring new customers is considerably more expensive than retaining existing ones, making churn prediction a critical business problem.

This project develops a machine learning pipeline to:

* Predict whether a customer is likely to churn.
* Identify the most important factors driving churn.
* Estimate the revenue at risk from potentially churning customers.
* Compare multiple machine learning models to determine the best-performing solution.

---

## Business Problem

Telecommunication companies lose substantial revenue when customers discontinue their services. Early identification of at-risk customers enables targeted retention strategies and reduces revenue loss.

### Objectives

* Predict customer churn before it occurs.
* Understand key drivers behind customer attrition.
* Quantify financial exposure through revenue-at-risk estimation.
* Enable targeted retention campaigns for high-value customers.

---

## Dataset

The project uses the **Telco Customer Churn Dataset**, which contains customer demographics, service usage information, and account details.

### Customer Information

* Gender
* Senior Citizen Status
* Partner Status
* Dependents

### Service Information

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Streaming Services

### Account Information

* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Tenure

### Target Variable

| Variable | Description           |
| -------- | --------------------- |
| Churn    | 0 = Customer Retained |
| Churn    | 1 = Customer Churned  |

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── customerChurn.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── XGB_final_model.pkl
│   └── encoders_data.pkl
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Preprocessing

Several preprocessing techniques were applied before model training.

### Missing Value Handling

* Identified invalid values in `TotalCharges`.
* Converted non-numeric entries into valid numerical format.
* Replaced missing values with `0.0`.

### Feature Transformation

To reduce skewness in the data:

* Applied a square-root transformation to `TotalCharges`.

### Categorical Encoding

Categorical features were converted into numerical representations using:

* Label Encoding

All encoders were saved using Pickle for future inference and deployment.

---

## Handling Class Imbalance

Customer churn datasets are naturally imbalanced, with significantly fewer churned customers than retained customers.

To address this issue:

* Applied **SMOTE (Synthetic Minority Oversampling Technique)** to the training data.
* Generated synthetic churn samples.
* Balanced the target classes.
* Improved model learning for minority-class predictions.

---

## Machine Learning Models Evaluated

Three classification models were evaluated using **5-Fold Cross Validation**.

### 1. Decision Tree

**Advantages**

* Easy to interpret.
* Fast training and inference.

### 2. Random Forest

**Advantages**

* Reduces overfitting.
* Handles noisy data effectively.
* Strong baseline performance.

### 3. XGBoost

**Advantages**

* Gradient boosting framework.
* Excellent predictive performance.
* Handles complex feature interactions.
* Widely used in industry competitions and production systems.

---

## Model Selection

Cross-validation was used to compare model performance.

### Evaluation Metrics

* Mean Accuracy
* Standard Deviation Across Folds

After comparison, **XGBoost** achieved the best overall performance and was selected as the final model.

---

## Results

### XGBoost Performance

| Metric              | Score  |
| ------------------- | ------ |
| Validation Accuracy | 77.71% |
| Test Accuracy       | 77.36% |

The final XGBoost model was trained on the complete SMOTE-balanced training dataset and saved for future deployment.

---

## Feature Importance Analysis

The XGBoost model identified the most influential factors contributing to customer churn.

### Top Churn Drivers

* Contract Type
* Tenure
* Monthly Charges
* Total Charges
* Internet Service
* Payment Method

These insights help businesses understand customer behavior and design more effective retention strategies.

---

## Revenue-at-Risk Framework

Beyond churn prediction, a Logistic Regression model was trained to estimate churn probabilities.

Revenue at Risk was calculated using:

```text
Revenue at Risk =
Monthly Charges × Probability of Churn
```

This metric estimates the expected monthly revenue loss associated with each customer.

### Example

| Monthly Charge | Churn Probability | Revenue at Risk |
| -------------- | ----------------- | --------------- |
| $100           | 90%               | $90             |

This enables businesses to prioritize customers based on both churn likelihood and financial impact.

---

## Key Business Insights

* High monthly-charge customers contribute disproportionately to revenue risk.
* Contract type is one of the strongest predictors of customer churn.
* Long-tenure customers are generally less likely to leave.
* Revenue-at-risk ranking helps prioritize retention efforts efficiently.
* Combining churn prediction with financial risk provides more actionable insights than churn prediction alone.

---

## Visualizations

The project includes multiple visual analyses to improve interpretability.

### Exploratory Data Analysis

* Numerical feature distributions
* Customer churn distribution
* Impact of SMOTE on class balance

### Model Evaluation

* Cross-validation comparison
* ROC Curves
* Precision-Recall Curves
* Confusion Matrix

### Explainability

* XGBoost Feature Importance
* Business-Oriented Churn Driver Analysis

### Business Analytics

* Revenue-at-Risk Dashboard
* High-Risk Customer Identification
* Revenue Exposure Analysis

---

## Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

### Model Persistence

* Pickle

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git

cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch the notebook:

```bash
jupyter notebook notebooks/customerChurn.ipynb
```

Or execute the training pipeline:

```bash
python src/train.py
```

---

## Future Improvements

Potential enhancements include:

* Hyperparameter optimization using Optuna.
* Probability calibration for improved churn probability estimates.
* SHAP-based explainability for individual predictions.
* FastAPI deployment for real-time inference.
* Docker containerization.
* Interactive business intelligence dashboard.
* Automated customer retention recommendation system.

---

## Conclusion

This project combines machine learning and business analytics to address a real-world customer retention problem. By integrating churn prediction with revenue-at-risk estimation, the solution moves beyond classification accuracy and delivers actionable business insights.

Organizations can use these predictions to proactively retain valuable customers, reduce churn, and minimize revenue loss through targeted intervention strategies.
