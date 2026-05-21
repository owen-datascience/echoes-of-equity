# feature_importance_trust.py
# This script trains a Random Forest classifier on a small synthetic
# acoustic feature dataset and then shows how to inspect feature importances.
# Every line is commented so that high school students can follow along.

# Import the pandas library to work with tabular data (CSV files).
import pandas as pd
# Import numpy for numerical operations (for sorting indices, etc.).
import numpy as np
# Import RandomForestClassifier model from scikit-learn.
from sklearn.ensemble import RandomForestClassifier
# Import train_test_split to make a simple train/test split.
from sklearn.model_selection import train_test_split
# Import accuracy_score to quickly measure model performance.
from sklearn.metrics import accuracy_score
# Import matplotlib.pyplot for plotting a bar chart of feature importances.
import matplotlib.pyplot as plt
# Import os to work with file paths.
import os

# Step 1: Load the dataset
# ----------------------------------

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_feature_importance_dataset.csv")
# Read the synthetic dataset from the CSV file into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows so we can see what the data looks like.
print("First 5 rows of the dataset:")
print(data.head())
print()

# Define the feature column names (input variables for the model).
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]
# X will store the feature values as a 2D table (rows = utterances, columns = features).
X = data[feature_columns]
# y will store the labels: 0 = neutral, 1 = trustworthy.
y = data["intent_label"]

# Step 2: Train/test split
# ----------------------------------

# Split data into training and test sets.
# test_size=0.3 means 30% of examples will be used for testing.
# random_state=0 makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Step 3: Train a Random Forest classifier
# ----------------------------------

# Create the RandomForestClassifier.
# n_estimators=200 means we use 200 decision trees in the forest.
rf = RandomForestClassifier(n_estimators=200, random_state=0)
# Fit (train) the model on the training data.
rf.fit(X_train, y_train)

# Step 4: Evaluate the model quickly with accuracy
# ----------------------------------

# Use the trained model to predict labels for the test set.
y_pred = rf.predict(X_test)
# Compute simple accuracy on the test set.
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy of the Random Forest:", test_accuracy)
print()

# Step 5: Get feature importances from the Random Forest
# ----------------------------------

# Random Forest gives an array of importance values, one for each feature.
importances = rf.feature_importances_

# Print the raw importance values along with their feature names.
print("Raw feature importances:")
for name, score in zip(feature_columns, importances):
    print(f"  {name}: {score}")
print()

# Step 6: Sort features by importance
# ----------------------------------

# Get the indices that would sort the importances array in descending order.
sorted_indices = np.argsort(importances)[::-1]

print("Features sorted by importance (highest to lowest):")
for idx in sorted_indices:
    feature_name = feature_columns[idx]
    importance_value = importances[idx]
    print(f"  {feature_name}: {importance_value}")
print()

# Step 7: Plot a simple bar chart of feature importances
# ----------------------------------

# Create a new figure for the bar chart.
plt.figure()
# Use the sorted indices so that the most important feature is on the left.
sorted_feature_names = [feature_columns[idx] for idx in sorted_indices]
sorted_importances = importances[sorted_indices]
# Create a bar chart of feature importances.
plt.bar(range(len(sorted_feature_names)), sorted_importances)
# Set x-axis tick positions.
plt.xticks(range(len(sorted_feature_names)), sorted_feature_names, rotation=45)
# Add a title so we know what the plot shows.
plt.title("Random Forest Feature Importances")
# Label the y-axis.
plt.ylabel("Importance")
# Adjust layout so labels do not overlap.
plt.tight_layout()
# Display the plot window.
plt.show()

# Step 8: Interpretation hint
# ----------------------------------

print("Interpretation hints:")
print("- Features with higher importance scores have more influence on the model's decisions.")
print("- In this synthetic dataset, pitch-related features (like mean_f0_hz) and voice quality")
print("  features (like hnr_db and cpp_db) should usually rank higher than others.")
print("- In a real research project, you would compare these results to domain knowledge")
print("  about how people change their voices when trying to sound trustworthy.")
