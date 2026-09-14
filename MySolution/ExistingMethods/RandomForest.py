import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)


le = LabelEncoder()
df['Speaker_Intent'] = le.fit_transform(df['Speaker_Intent'])


features = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                            'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
target = df['Speaker_Intent']


features = features.fillna(features.mean())


_eth_codes = LabelEncoder().fit_transform(df['Speaker_Ethnicity'])
_strata = target.values * 10 + _eth_codes 
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=_strata, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)


y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]


print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Random Forest AUC: {roc_auc_score(y_test, y_prob):.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
