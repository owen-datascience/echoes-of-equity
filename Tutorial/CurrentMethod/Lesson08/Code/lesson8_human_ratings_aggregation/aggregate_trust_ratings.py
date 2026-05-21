# aggregate_trust_ratings.py
# This script shows how to work with human trust ratings on speech clips.
# We start from a "long" ratings table (one row per [utterance, rater])
# and compute aggregated scores per utterance.
# Every line is commented so that high school students can follow it.

# Import pandas to load and work with the CSV file as a table.
import pandas as pd
# Import numpy for a few numerical operations.
import numpy as np
# Import matplotlib.pyplot to make a simple plot of rating distributions.
import matplotlib.pyplot as plt
# Import os to work with file paths.
import os

# Step 1: Load the rater-level ratings file
# ---------------------------------------------

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_ratings_long.csv")
# Read the synthetic ratings from the CSV file into a pandas DataFrame using absolute path.
ratings = pd.read_csv(csv_path)

# Print the first 10 rows so we can see what the data looks like.
print("First 10 rows of the ratings table:")
print(ratings.head(10))
print()

# Step 2: Basic info and sanity checks
# ---------------------------------------------

# Print some basic information: number of rows and columns.
print("Shape of ratings table (rows, columns):", ratings.shape)
# Show the column names to confirm what fields we have.
print("Columns:", list(ratings.columns))
print()

# Check how many unique utterances and raters are present.
n_utterances = ratings["utterance_id"].nunique()
n_raters = ratings["rater_id"].nunique()
print("Number of unique utterances:", n_utterances)
print("Number of unique raters:", n_raters)
print()

# Step 3: Aggregate ratings by utterance
# ---------------------------------------------

# Group the ratings by utterance_id so we can compute statistics per utterance.
grouped = ratings.groupby("utterance_id")

# Compute the mean rating, standard deviation, and number of ratings per utterance.
aggregated = grouped["trust_rating"].agg(
    mean_rating="mean",     # average rating across raters
    std_rating="std",       # how spread out the ratings are
    n_ratings="count"       # how many raters rated this utterance
).reset_index()

# Print the first few rows of the aggregated table.
print("Aggregated ratings per utterance:")
print(aggregated.head())
print()

# Step 4: Create a binary label from mean rating
# ---------------------------------------------

# We will define a simple rule:
# If mean_rating >= 4.0 (on a 1-7 scale), we call it "trustworthy" (1).
# Else we call it "neutral / low trust" (0).
threshold = 4.0
aggregated["intent_label"] = (aggregated["mean_rating"] >= threshold).astype(int)

# Print some counts of each label to check class balance.
print("Label counts based on mean_rating >= 4.0:")
print(aggregated["intent_label"].value_counts())
print()

# Step 5: Examine rating distributions
# ---------------------------------------------

# Make a histogram of mean ratings across utterances.
plt.figure()
plt.hist(aggregated["mean_rating"], bins=10)
plt.title("Histogram of mean trust ratings per utterance")
plt.xlabel("Mean rating (1-7 scale)")
plt.ylabel("Number of utterances")
plt.tight_layout()
plt.show()

# Make a scatter plot of (mean_rating vs std_rating).
plt.figure()
plt.scatter(aggregated["mean_rating"], aggregated["std_rating"])
plt.title("Mean rating vs. rating disagreement (std)")
plt.xlabel("Mean rating")
plt.ylabel("Standard deviation of ratings")
plt.tight_layout()
plt.show()

# Step 6: Save the utterance-level table for later modeling
# ---------------------------------------------

# Save the aggregated table to a new CSV file that can be used in modeling scripts.
aggregated.to_csv("trust_utterance_level_labels.csv", index=False)

print("Saved utterance-level labels to 'trust_utterance_level_labels.csv'.")
print()

# Step 7: Interpretation hints
# ---------------------------------------------

print("Interpretation hints:")
print("- The mean rating per utterance summarizes how trustworthy people felt the voice was.")
print("- The standard deviation (std_rating) shows how much raters disagreed.")
print("- The binary intent_label is a simplification; in real research, you might keep")
print("  the continuous rating or use more advanced methods to define categories.")
