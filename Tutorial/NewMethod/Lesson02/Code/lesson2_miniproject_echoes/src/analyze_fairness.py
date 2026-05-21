"""
Mini Project - Lesson 2: Baseline Fairness Analysis

This script:
1. Loads a small synthetic voice-intent dataset.
2. Shows label and demographic distributions.
3. Computes a simple "majority class" baseline by predicting the most common label.
4. Evaluates accuracy and recall overall and by ethnicity group.

Run with:
    python analyze_fairness.py
"""

# Import pandas - a library for working with data tables (like spreadsheets)
import pandas as pd
# Import Path from pathlib - helps us work with file paths in a cross-platform way
from pathlib import Path
# Import Counter from collections - helps us count things (not used in this code, but imported)
from collections import Counter

# Define a function that loads our dataset from a CSV file
def load_data():
    # Build the path to our data file by:
    # 1. Getting the current file's location (__file__)
    # 2. Going up one parent directory (parents[1])
    # 3. Then going into the "data" folder and finding "mini_trust_dataset.csv"
    data_path = Path(__file__).resolve().parents[1] / "data" / "mini_trust_dataset.csv"
    # Read the CSV file into a pandas DataFrame (a table of data)
    df = pd.read_csv(data_path)
    # Return the DataFrame so other functions can use it
    return df

# Define a function that prints out how the data is distributed
def print_distributions(df):
    # Print a header to organize our output
    print("=== Overall Label Distribution ===")
    # Count how many times each intent_label appears and show as percentages (normalize=True)
    # This tells us what percentage of our data has each label (like "trustworthy" vs "not trustworthy")
    print(df["intent_label"].value_counts(normalize=True))
    # Print a blank line then another header
    print("\n=== Ethnicity Distribution ===")
    # Count how many times each ethnicity appears and show as percentages
    # This tells us the diversity of our dataset
    print(df["ethnicity"].value_counts(normalize=True))

# Define a function that creates a simple baseline prediction model
def majority_baseline(df):
    # Create a prediction by always guessing the most common label in the dataset
    # mode() finds the most common value, [0] gets the first one (in case of ties)
    # This is called a "majority baseline" - the simplest possible prediction method
    majority_label = df["intent_label"].mode()[0]
    # Print what label we're going to predict every time
    print(f"\nMajority baseline label: {majority_label}")
    # Create a new column called "baseline_pred" where every row has the same prediction
    df["baseline_pred"] = majority_label
    # Return both the updated DataFrame and what the majority label was
    return df, majority_label

# Define a function that calculates how good our predictions are
def compute_metrics(df):
    # Import two functions from sklearn (a machine learning library):
    # - accuracy_score: tells us what percentage of predictions were correct
    # - recall_score: tells us what percentage of actual positive cases we correctly identified
    from sklearn.metrics import accuracy_score, recall_score

    # Print a header for the overall results
    print("\n=== Overall Performance (Majority Baseline) ===")
    # Calculate accuracy: compare real labels (intent_label) to our predictions (baseline_pred)
    acc = accuracy_score(df["intent_label"], df["baseline_pred"])
    # Calculate recall specifically for "trustworthy" cases
    # Recall tells us: of all the truly trustworthy cases, how many did we catch?
    rec = recall_score(df["intent_label"], df["baseline_pred"], pos_label="trustworthy")
    # Print the accuracy with 3 decimal places
    print(f"Accuracy: {acc:.3f}")
    # Print the recall with 3 decimal places
    print(f"Recall (trustworthy): {rec:.3f}")

    # Now check if our model is fair across different ethnic groups
    print("\n=== Performance by Ethnicity ===")
    # Loop through each ethnic group in our dataset
    # groupby("ethnicity") splits the data into separate groups by ethnicity
    # group = the name of the ethnicity, sub = the subset of data for that group
    for group, sub in df.groupby("ethnicity"):
        # Calculate accuracy for just this ethnic group
        acc_g = accuracy_score(sub["intent_label"], sub["baseline_pred"])
        # Calculate recall for just this ethnic group
        rec_g = recall_score(sub["intent_label"], sub["baseline_pred"], pos_label="trustworthy")
        # Print the results for this group
        # If accuracy/recall differs between groups, the model may be unfair
        print(f"{group}: Accuracy={acc_g:.3f}, Recall(trustworthy)={rec_g:.3f}")

# This special line checks if this file is being run directly (not imported)
if __name__ == "__main__":
    # Step 1: Load the data from the CSV file
    df = load_data()
    # Show the first few rows of data so we can see what it looks like
    print(df.head())
    # Step 2: Print statistics about how the data is distributed
    print_distributions(df)
    # Step 3: Create our simple baseline predictions
    # df gets updated with predictions, maj stores what label we're predicting
    df, maj = majority_baseline(df)
    # Step 4: Calculate and print how well our baseline model performs
    compute_metrics(df)
