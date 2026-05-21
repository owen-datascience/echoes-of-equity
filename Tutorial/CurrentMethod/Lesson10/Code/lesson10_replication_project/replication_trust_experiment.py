# replication_trust_experiment.py
# This script is a "mini replication" of the trust-detection pipeline.
# It uses a synthetic dataset that looks like an utterance-level version
# of the real paper's data and runs speaker-independent evaluation.
#
# The steps are:
#  1. Load the utterance-level dataset.
#  2. Build acoustic feature matrix (X), labels (y), and speaker groups.
#  3. Run Leave-One-Speaker-Out (LOSO) cross-validation.
#  4. Train Logistic Regression and Random Forest in each fold.
#  5. Compute accuracy, precision, recall, and F1 per speaker.
#  6. Average metrics and print a short text report.
#
# Every line is commented to help high school students understand.

import pandas as pd                      # For loading and handling CSV data.
import numpy as np                       # For numerical arrays and averages.
from sklearn.model_selection import LeaveOneGroupOut  # For LOSO CV.
from sklearn.linear_model import LogisticRegression    # Linear baseline.
from sklearn.ensemble import RandomForestClassifier    # Non-linear model.
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix)        # For metrics.
import os                                # For working with file paths.

# Step 1: Load the dataset
# ------------------------------

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "replication_trust_dataset.csv")
# Read in the utterance-level trust dataset using absolute path.
data = pd.read_csv(csv_path)

# Print the first few rows to see the structure.
print("First 6 rows of the replication dataset:")
print(data.head(6))
print()

# Show basic info (rows, columns).
print("Dataset shape (rows, columns):", data.shape)
# Show how many speakers and utterances we have.
print("Number of speakers:", data["speaker_id"].nunique())
print("Number of utterances:", data["utterance_id"].nunique())
print("Label counts (0 = neutral/low trust, 1 = trustworthy):")
print(data["intent_label"].value_counts())
print()

# Step 2: Build features, labels, and speaker groups
# ------------------------------

# Define which columns are acoustic features.
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]

# X will store feature values for each utterance.
X = data[feature_columns]
# y will store binary trust labels.
y = data["intent_label"]
# groups will store the speaker ID for each utterance (used for LOSO).
groups = data["speaker_id"]

# Step 3: Set up Leave-One-Speaker-Out cross-validation
# ------------------------------

logo = LeaveOneGroupOut()  # This object creates one fold per speaker.

# Lists to collect metrics across folds for both models.
log_accs, log_precs, log_recs, log_f1s = [], [], [], []
rf_accs, rf_precs, rf_recs, rf_f1s = [], [], [], []

# We will also collect all true labels and predictions across folds
# to build a global confusion matrix for each model.
all_true_log, all_pred_log = [], []
all_true_rf, all_pred_rf = [], []

fold_index = 0  # Counter for folds.

# Step 4: Loop over each speaker as the test set (LOSO)
# ------------------------------

for train_idx, test_idx in logo.split(X, y, groups):
    fold_index += 1  # Increase fold number.

    # Split the data into training and test subsets for this speaker.
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    test_speakers = groups.iloc[test_idx].unique()  # Speaker(s) in the test set.

    # --- Logistic Regression model ---
    log_model = LogisticRegression(max_iter=1000)  # Create the model.
    log_model.fit(X_train, y_train)               # Train it.
    y_pred_log = log_model.predict(X_test)        # Predict labels.

    # Compute metrics.
    log_acc = accuracy_score(y_test, y_pred_log)
    log_prec = precision_score(y_test, y_pred_log)
    log_rec = recall_score(y_test, y_pred_log)
    log_f1 = f1_score(y_test, y_pred_log)

    # Store metrics.
    log_accs.append(log_acc)
    log_precs.append(log_prec)
    log_recs.append(log_rec)
    log_f1s.append(log_f1)

    # Store true and predicted labels.
    all_true_log.extend(list(y_test))
    all_pred_log.extend(list(y_pred_log))

    # --- Random Forest model ---
    rf_model = RandomForestClassifier(n_estimators=250, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)

    rf_accs.append(rf_acc)
    rf_precs.append(rf_prec)
    rf_recs.append(rf_rec)
    rf_f1s.append(rf_f1)

    all_true_rf.extend(list(y_test))
    all_pred_rf.extend(list(y_pred_rf))

    # Print a brief per-fold summary.
    print(f"Fold {fold_index} - Test speaker(s): {list(test_speakers)}")
    print("  Logistic Regression: acc =", log_acc, ", F1 =", log_f1)
    print("  Random Forest:       acc =", rf_acc,  ", F1 =", rf_f1)
    print()

# Step 5: Convert lists to numpy arrays and compute averages
# ------------------------------

log_accs = np.array(log_accs)
log_precs = np.array(log_precs)
log_recs = np.array(log_recs)
log_f1s = np.array(log_f1s)

rf_accs = np.array(rf_accs)
rf_precs = np.array(rf_precs)
rf_recs = np.array(rf_recs)
rf_f1s = np.array(rf_f1s)

# Step 6: Build confusion matrices for both models
# ------------------------------

cm_log = confusion_matrix(all_true_log, all_pred_log)
cm_rf = confusion_matrix(all_true_rf, all_pred_rf)

# Step 7: Write a short replication-style report to a text file
# ------------------------------

# Open a text file in write mode.
with open("replication_report.txt", "w") as f:
    # Write dataset summary.
    f.write("Mini Replication Study: Voice Trust Classification (Synthetic Data)\n")
    f.write("----------------------------------------------------------------\n\n")
    f.write(f"Number of speakers: {data['speaker_id'].nunique()}\n")
    f.write(f"Number of utterances: {data['utterance_id'].nunique()}\n")
    f.write("Label counts (0=neutral/low trust, 1=trustworthy):\n")
    f.write(str(data['intent_label'].value_counts()) + "\n\n")

    # Logistic Regression results.
    f.write("=== Logistic Regression (LOSO) ===\n")
    f.write(f"Mean accuracy: {log_accs.mean():.3f} (std: {log_accs.std():.3f})\n")
    f.write(f"Mean precision: {log_precs.mean():.3f} (std: {log_precs.std():.3f})\n")
    f.write(f"Mean recall: {log_recs.mean():.3f} (std: {log_recs.std():.3f})\n")
    f.write(f"Mean F1-score: {log_f1s.mean():.3f} (std: {log_f1s.std():.3f})\n")
    f.write("Confusion matrix (rows=true, cols=pred):\n")
    f.write(str(cm_log) + "\n\n")

    # Random Forest results.
    f.write("=== Random Forest (LOSO) ===\n")
    f.write(f"Mean accuracy: {rf_accs.mean():.3f} (std: {rf_accs.std():.3f})\n")
    f.write(f"Mean precision: {rf_precs.mean():.3f} (std: {rf_precs.std():.3f})\n")
    f.write(f"Mean recall: {rf_recs.mean():.3f} (std: {rf_recs.std():.3f})\n")
    f.write(f"Mean F1-score: {rf_f1s.mean():.3f} (std: {rf_f1s.std():.3f})\n")
    f.write("Confusion matrix (rows=true, cols=pred):\n")
    f.write(str(cm_rf) + "\n\n")

    # Simple interpretation paragraph.
    f.write("Interpretation:\n")
    f.write("- Compare the mean F1-scores to decide which model performs better overall.\n")
    f.write("- Examine the confusion matrices to see whether models make more false positives\n")
    f.write("  (predicting trust when it is not there) or false negatives (missing trust).\n")
    f.write("- In a real replication of the paper, you would compare these average metrics\n")
    f.write("  to the reported results, and discuss similarities and differences.\n")

# Final message to the console.
print("Replication experiment finished. A detailed summary was written to 'replication_report.txt'.")
