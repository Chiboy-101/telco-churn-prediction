# Telco Customer Churn Prediction

Predicting customer churn is a critical task for telecommunications companies. This project demonstrates how to preprocess data, handle imbalanced datasets, train machine learning models, and evaluate their performance using **ROC-AUC** and other metrics.  

---

## 🚀 Project Overview

The goal is to predict whether a customer will churn (leave the company) based on demographic and service-related features.  
- **Dataset**: Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)  
- **Problem Type**: Binary classification  
- **Challenge**: Imbalanced classes (fewer churners than non-churners)  

---

## 📊 Dataset Description

| Column               | Description |
|----------------------|-------------|
| customerID           | Unique customer ID |
| gender               | Customer gender |
| SeniorCitizen        | Whether customer is senior |
| tenure               | Number of months customer has stayed |
| PhoneService         | Whether customer has phone service |
| MultipleLines        | Whether customer has multiple lines |
| InternetService      | Type of internet service |
| OnlineSecurity       | Whether customer has online security |
| OnlineBackup         | Whether customer has online backup |
| DeviceProtection     | Whether customer has device protection |
| TechSupport          | Whether customer has tech support |
| StreamingTV          | Whether customer streams TV |
| StreamingMovies      | Whether customer streams movies |
| Contract             | Contract type |
| PaperlessBilling     | Paperless billing flag |
| PaymentMethod        | Payment method |
| MonthlyCharges       | Monthly charges |
| TotalCharges         | Total charges |
| Churn                | Target variable (Yes/No) |

---

## 🛠 Data Preprocessing

- Dropped `customerID` column (irrelevant)
- Handled missing values (`TotalCharges`) and converted data types
- Replaced "No internet/phone service" with "No" to reduce noise
- Created a new feature: `avg_charge_per_month = TotalCharges / tenure`
- One-hot encoding for categorical variables
- Standard scaling of numeric features
- Handled class imbalance with **SMOTE** (Synthetic Minority Oversampling Technique)

---

## 🤖 Models Used

### Baseline Models

1. **Logistic Regression**  
2. **Decision Tree Classifier**  
3. **Random Forest Classifier**  

### Hyperparameter Tuning

- **Logistic Regression** tuned using `GridSearchCV`:
  - Parameters: `C`, `penalty`, `solver`, `class_weight`
---

## 📈 Evaluation Metrics

- **ROC-AUC**: Main metric for imbalanced classification  
- **Confusion Matrix**: True positives, false positives, etc.  
- **Precision, Recall, F1-Score**  

### Example Results: Logistic Regression (Tuned)

*CV ROC-AUC*: 0.8867

| Metric        | Value |
|---------------|-------|
| ROC-AUC       | 0.8291 |
| Accuracy      | 0.75   |
| Precision     | 0.53   |
| Recall        | 0.73   |
| F1-Score      | 0.61   |

**Confusion Matrix:**

[[1185 367]
[ 152 409]]


✅ High recall is ideal for detecting potential churners.  


---

## ⚙ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt

Then run the file:
```bash
python churn_prediction.py
