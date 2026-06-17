# Import necessary libraries for linear modeling and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

# 1. Load the speech dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
data = pd.read_csv(csv_path)

# 2. Target Encoding: Encode 'Neutral' as 0 and 'Trustworthy' as 1
encoder = LabelEncoder()
data['Speaker_Intent'] = encoder.fit_transform(data['Speaker_Intent'])

# 3. Clean Features: Remove identifiers and non-acoustic metadata
# We focus on Fundamental Frequency (F0), Jitter, Shimmer, and HNR as highlighted in the paper
X = data.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                       'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = data['Speaker_Intent']

# 4. Preprocessing: Address any Infinite or missing values in acoustic metrics
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

# 5. Joint Stratified Splitting: balance BOTH intent AND ethnicity in train/test.
# Stratifying on intent alone would let test-set ethnicity drift away from
# train-set proportions and inflate per-demographic accuracy variance.
_eth_codes = LabelEncoder().fit_transform(data['Speaker_Ethnicity'])
_strata = y.values * 10 + _eth_codes  # 6 cells: 2 intents x 3 ethnicities
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=_strata, random_state=42
)

# 6. Scaling: Logistic Regression is sensitive to the scale of features
# Scaling ensures convergence and accurate coefficient analysis
scaler = StandardScaler()
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled = scaler.transform(X_test)

# 7. Initialize and Train Logistic Regression
# 'liblinear' solver is efficient for this size of diverse speech dataset
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train_rescaled, y_train)

# 8. Predictions and Probability estimation
predictions = lr_model.predict(X_test_rescaled)
probabilities = lr_model.predict_proba(X_test_rescaled)[:, 1]

# 9. Output Metrics: Verify results against paper benchmarks
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"Logistic Regression AUC: {roc_auc_score(y_test, probabilities):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))