import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Get the directory where the script is actually sitting
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

METADATA = os.path.join(current_dir, "data/metadata_acoustic_prosody.csv")

ACOUSTIC_FEATURE_COLS = [
    "duration_sec",
    "f0_mean_hz",
    "f0_std_hz",
    "hnr_mean_db",
    "jitter_local",
    "cpp_mean_db",
]

df = pd.read_csv(METADATA)

# Remove rows with NaN
df = df.dropna(subset=ACOUSTIC_FEATURE_COLS)

X = df[ACOUSTIC_FEATURE_COLS].values
y = (df["intent"] == "trustworthy").astype(int).values

# Stratified split so neutral/trust have equal representation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# feature importance
for name, imp in zip(ACOUSTIC_FEATURE_COLS, clf.feature_importances_):
    print(f"{name:15s}: {imp:.4f}")
