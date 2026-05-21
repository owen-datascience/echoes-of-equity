# loso_cross_validation.py
# This script shows how to perform leave-one-speaker-out cross-validation (LOSO)
# on a small synthetic dataset of acoustic features for trust in voices.
# We will:
# - Load the dataset
# - Use speaker_id as the "group" for cross-validation
# - For each speaker: train on all other speakers, test on the held-out speaker
# - Compute accuracy, precision, recall, and F1-score for each fold
# - Print the average performance across all speakers

# Import pandas to load and work with the CSV dataset.
import pandas as pd
# Import numpy for basic numerical operations (for averaging metrics).
import numpy as np
# Import LeaveOneGroupOut to do leave-one-speaker-out cross-validation.
from sklearn.model_selection import LeaveOneGroupOut
# Import LogisticRegression as our classifier.
from sklearn.linear_model import LogisticRegression
# Import evaluation metrics: accuracy, precision, recall, F1-score.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_loso_dataset.csv")
# Read the synthetic dataset from the CSV file into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows so we can see what the data looks like.
print("First 8 rows of the dataset:")
print(data.head(8))
print()  # Blank line for readability.

# Define the list of feature column names we will use as inputs to the model.
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz", "hnr_db", "shimmer_db", "cpp_db"]
# X will contain only the feature values (as a 2D table).
X = data[feature_columns]
# y will contain the intent_label (0 = neutral, 1 = trustworthy) for each row.
y = data["intent_label"]
# groups will contain the speaker_id for each row, used for LOSO cross-validation.
groups = data["speaker_id"]

# Create a LeaveOneGroupOut object.
# This object knows how to create train/test splits that hold out one group at a time.
logo = LeaveOneGroupOut()

# Create Python lists to store the metrics for each speaker/fold.
accuracies = []   # This will store accuracy values.
precisions = []   # This will store precision values.
recalls = []      # This will store recall values.
f1_scores = []    # This will store F1-score values.

# Use the split method of LeaveOneGroupOut to loop over all possible splits.
# In each split, one unique speaker_id is used as the test group.
fold_index = 0  # We use fold_index to number the folds.
for train_idx, test_idx in logo.split(X, y, groups):
    # Increase the fold counter by 1.
    fold_index += 1

    # Use the indices to select training features and labels.
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    # Use the indices to select test features and labels.
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    # Also get the speaker IDs for the test set (for information).
    test_speakers = groups.iloc[test_idx].unique()

    # Create a LogisticRegression model with a higher max_iter value so it can converge.
    model = LogisticRegression(max_iter=1000)
    # Fit (train) the model on the training data (all speakers except the held-out one).
    model.fit(X_train, y_train)
    # Use the trained model to predict labels for the held-out test speaker.
    y_pred = model.predict(X_test)

    # Compute accuracy for this fold using the true and predicted labels.
    acc = accuracy_score(y_test, y_pred)
    # Compute precision for this fold (how many predicted trustworthy are correct).
    prec = precision_score(y_test, y_pred)
    # Compute recall for this fold (how many true trustworthy did we find).
    rec = recall_score(y_test, y_pred)
    # Compute F1-score for this fold (balance of precision and recall).
    f1 = f1_score(y_test, y_pred)

    # Append each metric to the corresponding list.
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    # Print the metrics for this fold so we can see performance per speaker.
    print(f"Fold {fold_index} - Test speaker(s): {list(test_speakers)}")
    print("  Accuracy:", acc)
    print("  Precision:", prec)
    print("  Recall:", rec)
    print("  F1-score:", f1)
    print()

# After the loop, convert the metric lists into numpy arrays for easier averaging.
accuracies = np.array(accuracies)
precisions = np.array(precisions)
recalls = np.array(recalls)
f1_scores = np.array(f1_scores)

# Print the average metrics across all folds (all speakers).
print("=== Average performance across all speakers (LOSO) ===")
print("Mean accuracy:", accuracies.mean())
print("Mean precision:", precisions.mean())
print("Mean recall:", recalls.mean())
print("Mean F1-score:", f1_scores.mean())
print()

# Print also the standard deviation to see how much the metrics vary across speakers.
print("Standard deviation of accuracy:", accuracies.std())
print("Standard deviation of precision:", precisions.std())
print("Standard deviation of recall:", recalls.std())
print("Standard deviation of F1-score:", f1_scores.std())
print()

# Final interpretation hint for the student.
print("Interpretation hints:")
print("- Each fold corresponds to leaving out one speaker for testing.")
print("- The average metrics tell you how well the model generalizes to new speakers.")
print("- The standard deviation tells you how performance varies from speaker to speaker.")
