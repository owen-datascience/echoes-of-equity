import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
data = pd.read_csv(csv_path)

encoder = LabelEncoder()
data['Speaker_Intent'] = encoder.fit_transform(data['Speaker_Intent'])

X = data.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                       'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = data['Speaker_Intent']

X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

_eth_codes = LabelEncoder().fit_transform(data['Speaker_Ethnicity'])
_strata = y.values * 10 + _eth_codes  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=_strata, random_state=42
)


scaler = StandardScaler()
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled = scaler.transform(X_test)

lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train_rescaled, y_train)

predictions = lr_model.predict(X_test_rescaled)
probabilities = lr_model.predict_proba(X_test_rescaled)[:, 1]

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"Logistic Regression AUC: {roc_auc_score(y_test, probabilities):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
