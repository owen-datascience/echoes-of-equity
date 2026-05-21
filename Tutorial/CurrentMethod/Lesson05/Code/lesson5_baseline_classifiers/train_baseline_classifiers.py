# train_baseline_classifiers.py
# This script trains two simple baseline classifiers:
# 1) Logistic Regression
# 2) Random Forest
# on a small synthetic dataset of acoustic features.
# The goal is to practice the full supervised learning pipeline:
# - load data
# - split into train and test sets
# - fit models
# - evaluate with accuracy, precision, recall, and F1-score

# Import pandas for working with the CSV file as a DataFrame.
import pandas as pd
# Import train_test_split to divide data into training and testing sets.
from sklearn.model_selection import train_test_split
# Import LogisticRegression model.
from sklearn.linear_model import LogisticRegression
# Import RandomForestClassifier model.
from sklearn.ensemble import RandomForestClassifier
# Import evaluation metrics to measure model performance.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_baseline_dataset.csv")
# Read the synthetic dataset from the CSV file using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows so you can see what the data looks like.
print("First 5 rows of the dataset:")
print(data.head())
print()

# Define the feature columns (inputs to the model).
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz", "hnr_db", "shimmer_db", "cpp_db"]
# X contains only the feature values for all rows.
X = data[feature_columns]
# y contains the target label for each row: 0 = neutral, 1 = trustworthy.
y = data["intent_label"]

# Split the data into training and test sets.
# test_size=0.3 means 30% of the data will be used for testing.
# random_state=0 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# -----------------------------
# Logistic Regression baseline
# -----------------------------

# Create a LogisticRegression model object.
log_reg = LogisticRegression(max_iter=1000)
# Fit (train) the logistic regression model on the training data.
log_reg.fit(X_train, y_train)
# Use the trained model to make predictions on the test set.
y_pred_log = log_reg.predict(X_test)

# Compute evaluation metrics for logistic regression.
acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)

# Print the results for logistic regression.
print("=== Logistic Regression Results ===")
print("Accuracy:", acc_log)
print("Precision:", prec_log)
print("Recall:", rec_log)
print("F1-score:", f1_log)
print("Confusion matrix:")
print(cm_log)
print()

# --------------------------
# Random Forest baseline
# --------------------------

# Create a RandomForestClassifier model object.
# n_estimators=100 means we use 100 decision trees.
rf = RandomForestClassifier(n_estimators=100, random_state=0)
# Fit (train) the random forest model on the training data.
rf.fit(X_train, y_train)
# Use the trained model to make predictions on the test set.
y_pred_rf = rf.predict(X_test)

# Compute evaluation metrics for random forest.
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Print the results for random forest.
print("=== Random Forest Results ===")
print("Accuracy:", acc_rf)
print("Precision:", prec_rf)
print("Recall:", rec_rf)
print("F1-score:", f1_rf)
print("Confusion matrix:")
print(cm_rf)
print()

# Optional: print a detailed classification report for each model.
print("Classification report for Logistic Regression:")
print(classification_report(y_test, y_pred_log))
print()

print("Classification report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
print()

# Print a final hint for interpretation.
print("Interpretation hints:")
print("- Compare accuracy, precision, recall, and F1-score between the two models.")
print("- Notice which model performs slightly better on this synthetic dataset.")
print("- Remember that these are simple baselines; the goal is to understand the workflow.")
