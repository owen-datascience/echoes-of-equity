# trust_eda_plots.py
# This script performs simple Exploratory Data Analysis (EDA)
# on a small synthetic dataset that mimics a trust-in-voice study.
# You will see how to:
# - Load the dataset
# - Inspect basic information about the data
# - Compare feature distributions for neutral vs trustworthy intent
# - Make simple histograms using matplotlib

# Import pandas for working with tabular data (DataFrames).
import pandas as pd
# Import matplotlib.pyplot for creating basic plots.
import matplotlib.pyplot as plt
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_eda_dataset.csv")
# Read the CSV dataset into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows so you can see what the table looks like.
print("First 10 rows of the dataset:")
print(data.head(10))
print()

# Print the shape of the DataFrame: (number_of_rows, number_of_columns).
print("Dataset shape (rows, columns):")
print(data.shape)
print()

# Print the column names so you know what fields are available.
print("Column names:")
print(list(data.columns))
print()

# Show how many utterances belong to each intent label (0 = neutral, 1 = trustworthy).
print("Class distribution for intent_label:")
print(data["intent_label"].value_counts())
print()

# Compute summary statistics (min, max, mean, etc.) for numeric features.
numeric_cols = ["duration_seconds", "mean_f0_hz", "sd_f0_hz", "hnr_db"]
print("Summary statistics for numeric features:")
print(data[numeric_cols].describe())
print()

# Group by intent_label and compute mean of each numeric feature.
print("Mean feature values by intent_label (0 = neutral, 1 = trustworthy):")
print(data.groupby("intent_label")[numeric_cols].mean())
print()

# Now we will create a histogram of mean_f0_hz separated by intent_label.
# This lets you visually compare pitch distributions for neutral vs trustworthy speech.
neutral = data[data["intent_label"] == 0]
trustworthy = data[data["intent_label"] == 1]

# Create a new figure for the first histogram plot.
plt.figure()
# Plot histogram of mean_f0_hz for neutral utterances.
plt.hist(neutral["mean_f0_hz"], bins=10, alpha=0.7, label="Neutral")
# Plot histogram of mean_f0_hz for trustworthy utterances.
plt.hist(trustworthy["mean_f0_hz"], bins=10, alpha=0.7, label="Trustworthy")
# Add a title to the plot so we know what it shows.
plt.title("Histogram of mean_f0_hz by intent_label")
# Label the x-axis to show the feature name.
plt.xlabel("mean_f0_hz (Hz)")
# Label the y-axis to show this is a count of utterances.
plt.ylabel("Count of utterances")
# Add a legend to explain the colors.
plt.legend()
# Show the plot on the screen.
plt.show()

# Next, create a histogram for hnr_db to compare voice clarity across intents.
plt.figure()
# Plot histogram of hnr_db for neutral utterances.
plt.hist(neutral["hnr_db"], bins=10, alpha=0.7, label="Neutral")
# Plot histogram of hnr_db for trustworthy utterances.
plt.hist(trustworthy["hnr_db"], bins=10, alpha=0.7, label="Trustworthy")
# Add a title to explain what the histogram shows.
plt.title("Histogram of hnr_db by intent_label")
# Label the x-axis with the feature name.
plt.xlabel("hnr_db")
# Label the y-axis with the word 'Count'.
plt.ylabel("Count of utterances")
# Add a legend to distinguish neutral and trustworthy bars.
plt.legend()
# Show the plot on the screen.
plt.show()

# Finally, print a brief interpretation hint for the student.
print("Interpretation hints:")
print("- Look at whether the trustworthy histograms are shifted slightly")
print("  toward higher pitch (mean_f0_hz) and higher HNR values compared")
print("  to neutral. This mimics patterns that might appear in a real study.")
