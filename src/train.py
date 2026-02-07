# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# Handle missing values and datatypes
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})
df.replace(["No internet service", "No phone service"], "No", inplace=True)

# Create new feature
df["avg_charge_per_month"] = df["TotalCharges"] / np.where(
    df["tenure"] == 0, 1, df["tenure"]
)

# Check for outliers
numeric_col = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_col:
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.show()

# One-Hot Encoding instead of Label Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split the dataset
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Scale numeric columns
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
X_train = X_train.copy()
X_test = X_test.copy()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Try baseline models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    " Random Forest Classifier": RandomForestClassifier(random_state=42),
}

# Loop through each model and get its predictions and evaluations
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"{name}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Hyperparameter tuning for Logistic Regression
log_reg = LogisticRegression(max_iter=3000, random_state=42)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "class_weight": ["balanced", {0: 1, 1: 3}],
}

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    n_jobs=1,
)

grid_search.fit(X_train, y_train)

# Check best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_1 = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n Tuned Model")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(confusion_matrix(y_test, y_pred_1))
print(classification_report(y_test, y_pred_1))

# Save the model
with open("best_logistic_regression.pkl", "wb") as f:
    pickle.dump(best_model, f)
