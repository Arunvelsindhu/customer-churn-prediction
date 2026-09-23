# ⚡ ChurnSense AI — Customer Churn Prediction

AI-powered web app that predicts whether a telecom customer is likely to churn, built with Python, Scikit-learn, and Streamlit.

## Overview

Customer churn — when a customer stops using a company's service — costs businesses significant revenue. This project analyzes the IBM Telco Customer Churn dataset to identify churn drivers and predicts churn risk for individual customers through an interactive web interface.

## Key Findings (EDA)

- **Contract type** is the strongest churn driver — month-to-month customers churn far more than those on annual/biennial contracts
- **Tenure** matters heavily — new customers (first few months) are highest-risk
- **Fiber optic internet** and **Electronic check** payment correlate with higher churn
- Customers with more add-on services (security, backup, streaming) churn less — they're more "locked in"

## Model Performance

Three models were trained and compared: Logistic Regression, Decision Tree, and Random Forest — all with `class_weight='balanced'` to address the dataset's ~73/27 class imbalance.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.783 | 0.614 |
| **Decision Tree (selected)** | **0.739** | **0.505** | **0.821** | **0.626** |
| Random Forest | 0.760 | 0.536 | 0.711 | 0.611 |

**Decision Tree** was selected as the final model — it had the highest recall on the churn class, meaning it catches the most actual churners. In a retention context, missing a real churner (false negative) is more costly than a false alarm, so recall was prioritized over raw accuracy.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit · Joblib

## Project Structure