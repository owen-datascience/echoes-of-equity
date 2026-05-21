# full_trust_pipeline_loso.py
# This script shows a complete end-to-end pipeline for a simple trust-detection task.
# It starts from an utterance-level dataset that includes:
#  - Speaker IDs
#  - Acoustic features
#  - Aggregated human trust ratings (mean and std)
#  - A binary trust label (intent_label)
# The script then:
#  1. Loads the dataset
#  2. Inspects basic statistics
#  3. Builds feature, label, and group arrays
#  4. Runs Leave-One-Speaker-Out (LOSO) cross-validation
#  5. Trains Logistic Regression and Random Forest models
#  6. Computes average metrics (accuracy, precision, recall, F1)
#  7. Prints a short summary comparing the models
# Every block is heavily commented so high school students can follow each step.

# Import pandas to load the CSV file and handle tables.
import pandas as pd
# Import numpy for numerical arrays and averaging values.
import numpy as np
# Import LeaveOneGroupOut for speaker-independent cross-validation.
from sklearn.model_selection import LeaveOneGroupOut
# Import the LogisticRegression classifier.
from sklearn.linear_model import LogisticRegression
# Import the RandomForestClassifier.
from sklearn.ensemble import RandomForestClassifier
# Import the standard evaluation metrics.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Import os to work with file paths.
import os

# Step 1: Load the full-pipeline dataset
# -----------------------------------------

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "synthetic_trust_full_pipeline_dataset.csv")
# Read the synthetic dataset from the CSV file into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Print the first 8 rows so we can see the structure of the data.
print("First 8 rows of the dataset:")
print(data.head(8))
print()

# Print basic information about the dataset: number of rows and columns.
print("Shape of dataset (rows, columns):", data.shape)
# Print the column names to understand what fields are present.
print("Columns:", list(data.columns))
print()

# Step 2: Basic counts and sanity checks
# -----------------------------------------

# Count how many unique speakers we have.
n_speakers = data["speaker_id"].nunique()
# Count how many utterances we have.
n_utterances = data["utterance_id"].nunique()
# Count how many positive and negative labels we have.
label_counts = data["intent_label"].value_counts()

print("Number of unique speakers:", n_speakers)
print("Number of unique utterances:", n_utterances)
print("Label counts (0 = neutral/low trust, 1 = trustworthy):")
print(label_counts)
print()

# Step 3: Build feature matrix (X), label vector (y), and group vector (speaker IDs)
# ----------------------------------------------------------------------------------

# Define the acoustic feature columns we will use as model inputs.
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]

# X is our feature matrix: rows = utterances, columns = acoustic features.
X = data[feature_columns]
# y is our label vector: 0 = neutral/low trust, 1 = trustworthy.
y = data["intent_label"]
# groups is our group vector used for LOSO: each value is a speaker_id.
groups = data["speaker_id"]

# Step 4: Set up LeaveOneGroupOut cross-validation (LOSO)
# -------------------------------------------------------

# Create a LeaveOneGroupOut object from scikit-learn.
logo = LeaveOneGroupOut()

# Create lists to store metrics for each model across folds.
log_accs, log_precs, log_recs, log_f1s = [], [], [], []
rf_accs, rf_precs, rf_recs, rf_f1s = [], [], [], []

# Keep track of which fold we are on (for printing).
fold_index = 0

# Step 5: Loop over LOSO folds
# -------------------------------------------------------

# The split method creates one train/test split per speaker.
for train_idx, test_idx in logo.split(X, y, groups):
    # Increase fold_index so we can label this fold.
    fold_index += 1

    # Use the indices to get training and test data for this fold.
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    # Get the speaker IDs that are in the test set for this fold.
    test_speakers = groups.iloc[test_idx].unique()

    # --- Logistic Regression model ---
    # Create a LogisticRegression model with a higher max_iter so it converges.
    log_model = LogisticRegression(max_iter=1000)
    # Fit the model on the training data.
    log_model.fit(X_train, y_train)
    # Predict labels for the test data.
    y_pred_log = log_model.predict(X_test)

    # Compute evaluation metrics for Logistic Regression on this fold.
    log_acc = accuracy_score(y_test, y_pred_log)
    log_prec = precision_score(y_test, y_pred_log)
    log_rec = recall_score(y_test, y_pred_log)
    log_f1 = f1_score(y_test, y_pred_log)

    # Store the metrics in their lists.
    log_accs.append(log_acc)
    log_precs.append(log_prec)
    log_recs.append(log_rec)
    log_f1s.append(log_f1)

    # --- Random Forest model ---
    # Create a RandomForestClassifier with 200 trees.
    rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
    # Fit the Random Forest on the training data.
    rf_model.fit(X_train, y_train)
    # Predict labels for the test data.
    y_pred_rf = rf_model.predict(X_test)

    # Compute evaluation metrics for Random Forest on this fold.
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)

    # Store the metrics.
    rf_accs.append(rf_acc)
    rf_precs.append(rf_prec)
    rf_recs.append(rf_rec)
    rf_f1s.append(rf_f1)

    # Print per-fold summary for both models.
    print(f"Fold {fold_index} - Test speaker(s): {list(test_speakers)}")
    print("  Logistic Regression - acc:", log_acc, "prec:", log_prec, "rec:", log_rec, "f1:", log_f1)
    print("  Random Forest        - acc:", rf_acc, "prec:", rf_prec, "rec:", rf_rec, "f1:", rf_f1)
    print()

# Step 6: Convert metric lists to numpy arrays for averaging
# -------------------------------------------------------

log_accs = np.array(log_accs)
log_precs = np.array(log_precs)
log_recs = np.array(log_recs)
log_f1s = np.array(log_f1s)

rf_accs = np.array(rf_accs)
rf_precs = np.array(rf_precs)
rf_recs = np.array(rf_recs)
rf_f1s = np.array(rf_f1s)

# Step 7: Print average performance across all speakers for each model
# -------------------------------------------------------

print("=== Logistic Regression (LOSO) - Average performance across speakers ===")
print("Mean accuracy:", log_accs.mean(), "  Std accuracy:", log_accs.std())
print("Mean precision:", log_precs.mean(), " Std precision:", log_precs.std())
print("Mean recall:", log_recs.mean(), "    Std recall:", log_recs.std())
print("Mean F1-score:", log_f1s.mean(), "  Std F1-score:", log_f1s.std())
print()

print("=== Random Forest (LOSO) - Average performance across speakers ===")
print("Mean accuracy:", rf_accs.mean(), "  Std accuracy:", rf_accs.std())
print("Mean precision:", rf_precs.mean(), " Std precision:", rf_precs.std())
print("Mean recall:", rf_recs.mean(), "    Std recall:", rf_recs.std())
print("Mean F1-score:", rf_f1s.mean(), "  Std F1-score:", rf_f1s.std())
print()

# Step 8: Simple comparison summary
# -------------------------------------------------------

print("Summary comparison (higher is better):")
print("- Logistic Regression mean F1:", log_f1s.mean())
print("- Random Forest mean F1:", rf_f1s.mean())
print("If Random Forest has higher mean F1, it is capturing more complex patterns")
print("in the acoustic features for this synthetic trust dataset.")
