# explore_trust_dataset_structure.py
# This script shows how to explore a small synthetic version
# of the "trustworthy intent in speech" dataset.
# The goal is to understand how speakers, utterances, and labels
# are organized in a table, similar to the real research dataset.

# Import the pandas library so we can read and analyze the CSV file.
import pandas as pd
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_dataset_structure.csv")
# Read the synthetic dataset into a pandas DataFrame using absolute path.
# Each row is one utterance (one spoken sentence).
data = pd.read_csv(csv_path)

# Print the first few rows so we can see what the columns look like.
print("First 8 rows of the dataset:")
print(data.head(8))
print()  # Print a blank line for readability.

# Print the column names to see what information is available.
print("Column names:")
print(list(data.columns))
print()

# Print the shape of the dataset (number of rows and columns).
print("Dataset shape (rows, columns):")
print(data.shape)
print()

# Show the unique speaker IDs to see how many speakers we have.
print("Unique speaker IDs:")
print(sorted(data["speaker_id"].unique()))
print()

# Count how many utterances each speaker has.
# This helps us see if every speaker has the same number of recordings.
print("Number of utterances per speaker:")
print(data["speaker_id"].value_counts().sort_index())
print()

# Look at how many examples we have for each intent label overall.
# intent_label: 0 = neutral, 1 = trustworthy.
print("Overall distribution of intent labels (0 = neutral, 1 = trustworthy):")
print(data["intent_label"].value_counts())
print()

# Now group by speaker and intent_label to check that each speaker
# has both neutral and trustworthy utterances.
counts_by_speaker_intent = data.groupby(["speaker_id", "intent_label"]).size()
print("Number of utterances per speaker and intent label:")
print(counts_by_speaker_intent)
print()

# Examine how speakers are distributed across age groups, sex, and ethnicity.
print("Speakers by age_group:")
print(data.groupby("speaker_id")["age_group"].first().value_counts())
print()

print("Speakers by sex:")
print(data.groupby("speaker_id")["sex"].first().value_counts())
print()

print("Speakers by ethnicity:")
print(data.groupby("speaker_id")["ethnicity"].first().value_counts())
print()

# Compute simple summary statistics for the numeric columns.
# This gives us min, max, mean, etc., for duration and mean_f0_hz.
print("Summary statistics for numeric features:")
print(data[["duration_seconds", "mean_f0_hz"]].describe())
print()

# Finally, print a short interpretation hint for the student.
print("Interpretation hints:")
print("- Each row is one utterance with metadata (speaker_id, age_group, etc.)")
print("  and labels (intent_label) plus simple acoustic features.")
print("- Check that each speaker appears in both classes (neutral and trustworthy).")
print("- Notice how demographic information is stored as separate columns,")
print("  which lets us later analyze performance by subgroup.")
