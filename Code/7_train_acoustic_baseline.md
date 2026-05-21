Great — now that you have **metadata_acoustic_prosody.csv** with REAL prosodic features (F0, jitter, HNR, CPP, duration), you are ready to start the most important phase of your ISEF project:

---

# 🎯 **What you should do next**

There are **three essential steps**, and I recommend doing them in the following order:

---

# ✅ **STEP 1 — Train three baseline models and compare them**

This is *critical* for your scientific contribution.

Train:

---

### **Model A — Acoustic-only ML model**

(e.g., Random Forest or XGBoost)

Input features:

```
duration_sec
f0_mean_hz
f0_std_hz
hnr_mean_db
jitter_local
cpp_mean_db
```

Expected:

* Accuracy ≈ 55–70%
* Trustworthy recall > 0
* Provides interpretability
* Great for ISEF because you can show **which features matter most** (e.g., F0_std, CPP)

---

### **Model B — WavLM-only deep model**

(no prosody features)

Expected:

* Accuracy ≈ 50–60%
* This shows pretrained models struggle with trust cues

---

### **Model C — Hybrid Model (WavLM + acoustic features)**

This will be your **main winning model**.

Expected:

* Accuracy ≈ 65–80%
* Better trust recall
* Justify your scientific claim:

  > “Prosody features improve trust detection beyond pretrained speech models.”

---

## ✔ This comparison (A vs B vs C) is REQUIRED for any ISEF-level project.

It shows:

* You tried multiple approaches
* You identified *why* they worked
* Your conclusion is based strictly on experiments (not speculation)

---

# 📌 Python: Train Model A (Acoustic-Only Random Forest)

Before training the hybrid model, **train the acoustic-only baseline**.
It’s simple and extremely informative.

### Save as `train_acoustic_baseline.py`:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

METADATA = "data/metadata_acoustic_prosody.csv"

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
```

This gives:

* first model
* first accuracy benchmark
* interpretable feature importances

---

# 🎯 **STEP 2 — Retrain the hybrid deep model using the new features**

Next, update your hybrid WavLM model:

```python
METADATA_CSV = r"data/metadata_acoustic_prosody.csv"

ACOUSTIC_FEATURE_COLS = [
    "duration_sec",
    "f0_mean_hz",
    "f0_std_hz",
    "hnr_mean_db",
    "jitter_local",
    "cpp_mean_db",
]
```

Then run your hybrid model script again.

**Why this matters:**
When the hybrid model beats the acoustic-only model, you can say:

> “Deep-learned representation + prosody cues outperform either one alone.”
> This is textbook ISEF originality.

---

# 🎯 **STEP 3 — Perform fairness & demographic analysis**

Your dataset includes:

* **ethnicity:** white / black / south_asian
* **age_group:** older / younger
* **sex:** male / female

After training the hybrid model, compute accuracy for each group.

Example:

```
Black speakers:      0.62
White speakers:      0.74
South Asian:         0.68
Older adults:        0.70
Younger adults:      0.66
Male:                0.68
Female:              0.72
```

These differences become the core of your ISEF research paper.

---

# 🧠 **Interpretation for your ISEF paper**

After completing Steps 1–3 you can answer:

### 1. Are trust cues universal or speaker-dependent?

(You will see high variance by speaker.)

### 2. Does prosody improve trust recognition over WavLM-only?

(Test hybrid vs WavLM-only accuracy.)

### 3. Are there demographic fairness differences?

(Accuracy gaps across ethnicity / age / sex.)

### 4. Which features matter most?

(Random Forest feature importance + ablations.)

---

# 🚀 **Once you complete Step 1 (acoustic-only model), send me your results.**

I will help you:

* interpret them,
* build plots,
* improve the hybrid model,
* prepare your ISEF graphs, tables, and text.

👉 **Please run `train_acoustic_baseline.py` and paste the results.**
