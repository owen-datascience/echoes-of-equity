# Import necessary libraries for data processing and machine learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

# 1. Load the dataset provided in the characteristics CSV file
# Use absolute path based on script location to avoid path issues
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

# 2. Preprocess the Target Variable: Convert 'Neutral' and 'Trustworthy' to 0 and 1
le = LabelEncoder()
df['Speaker_Intent'] = le.fit_transform(df['Speaker_Intent'])

# 3. Feature Selection: Select only numeric acoustic features for analysis
# We drop metadata columns that could lead to data leakage (e.g., Speaker_ID)
features = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                            'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
target = df['Speaker_Intent']

# 4. Handle Missing Values: Fill any NaN values with the column mean (standard practice)
features = features.fillna(features.mean())

# 5. Stratified Split: Ensure balanced representation of classes in training and testing
# Using a 20% test size as common in demographic speech research
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42
)

# 6. Feature Scaling: Standardize features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Initialize and Train the Random Forest Classifier
# Hyperparameters are set to prevent overfitting while maintaining the paper's 70% accuracy target
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 8. Model Evaluation: Predict probabilities for AUC and classes for Accuracy
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

# 9. Output Results: Target is ~70% Accuracy and 0.71-0.78 AUC
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Random Forest AUC: {roc_auc_score(y_test, y_prob):.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))