# train_logistic_regression.py
# This script trains a simple Logistic Regression model
# to predict whether a voice sample is Neutral (0)
# or Trustworthy (1) based on three acoustic features.

# Import the pandas library for reading the CSV file as a table.
import pandas as pd
# Import the train_test_split helper function to split data into train and test sets.
from sklearn.model_selection import train_test_split
# Import the LogisticRegression model from scikit-learn.
from sklearn.linear_model import LogisticRegression
# Import the accuracy_score function to measure how well the model works.
from sklearn.metrics import accuracy_score
# Import os to work with file paths.
import os

# Get the directory where this script is located.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the CSV file (assumes it's in the same directory as the script).
csv_path = os.path.join(script_dir, "mini_voice_trust_dataset.csv")
# Read the CSV dataset file into a pandas DataFrame using absolute path.
data = pd.read_csv(csv_path)

# Select the feature columns (inputs) that the model will learn from.
X = data[["duration_seconds", "mean_pitch_hz", "hnr_db"]]
# Select the target column (output label) that we want the model to predict.
y = data["intent_label"]

# Split the data into training and test sets so we can evaluate generalization.
# test_size=0.3 means 30% of the data is used for testing.
# random_state=0 keeps the split the same every time we run the script.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Create a LogisticRegression model object.
model = LogisticRegression()
# Fit (train) the model on the training data.
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set.
y_pred = model.predict(X_test)

# Calculate the accuracy score by comparing predictions to true labels.
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy so the student can see how well the model performs.
print("Test accuracy:", accuracy)

# Print the model coefficients so the student can see feature importance direction.
print("Model coefficients (for [duration_seconds, mean_pitch_hz, hnr_db]):")
print(model.coef_)
# Print the intercept term of the model.
print("Model intercept:")
print(model.intercept_)
