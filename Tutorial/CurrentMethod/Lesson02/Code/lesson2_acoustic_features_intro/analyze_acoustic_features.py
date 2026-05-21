# analyze_acoustic_features.py
# This script helps you explore a small synthetic dataset of acoustic features.
# Each row represents one spoken sentence (an "utterance").
# The features describe properties of the voice, such as pitch and voice quality.
# The goal is to see how these features differ between neutral and trustworthy intent.

# Import the pandas library so we can work with the CSV file as a table.
import pandas as pd
# Import the numpy library for basic numerical operations (we will use it later if needed).
import numpy as np
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_acoustic_features.csv")
# Read the CSV dataset into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows so we can see how the data looks.
print("First 5 rows of the dataset:")
print(data.head())
print()  # Print a blank line for readability.

# Print the shape of the dataset: (number_of_rows, number_of_columns).
print("Dataset shape (rows, columns):")
print(data.shape)
print()

# Separate the feature columns (inputs) from the label column (output).
feature_columns = ["mean_f0_hz", "sd_f0_hz", "hnr_db", "shimmer_db", "cpp_db", "duration_seconds"]
# X will contain only the acoustic feature values.
X = data[feature_columns]
# y will contain the intent label: 0 = neutral, 1 = trustworthy.
y = data["intent_label"]

# Print basic statistics for each feature to see their ranges and averages.
print("Basic statistics for each acoustic feature:")
print(X.describe())
print()

# Group the data by intent_label and compute the mean of each feature per group.
grouped_means = data.groupby("intent_label")[feature_columns].mean()
# Print the group means to see how neutral and trustworthy voices differ.
print("Mean feature values by intent_label (0 = neutral, 1 = trustworthy):")
print(grouped_means)
print()

# Compute the difference in means between trustworthy (1) and neutral (0).
# This shows how much each feature tends to increase or decrease with trustworthy intent.
mean_diff = grouped_means.loc[1] - grouped_means.loc[0]
print("Difference in mean feature values (trustworthy - neutral):")
print(mean_diff)
print()

# Compute a simple correlation matrix for the features.
# Correlation shows how features move together (positive, negative, or no relation).
print("Correlation matrix of acoustic features:")
print(X.corr())
print()

# Finally, print a short interpretation hint for the student.
print("Interpretation hint:")
print("- Look at which features have higher average values for trustworthy intent.")
print("- For example, if mean_f0_hz is higher for trustworthy,")
print("  that means our synthetic data assumes trustworthy speech has higher pitch.")
