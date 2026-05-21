# Tutorial 08 — Comparing Models & Next Steps

You've built four models. Let's compare them, talk about what to learn next, and (since this is the **Echoes of Equity** project) finally connect the work back to fairness.

---

## Putting it all on one table

Here are typical results (yours will differ by a few points):

| Model | Type | Accuracy | AUC | Trains in |
|-------|------|----------|-----|-----------|
| Logistic Regression | Linear, classical | ~0.65 | ~0.70 | < 1 sec |
| Random Forest | Tree ensemble, classical | ~0.70 | ~0.75 | ~1 sec |
| ANN (Dense) | Deep learning | ~0.72 | ~0.79 | ~5 sec |
| 1D CNN | Deep learning | ~0.71 | ~0.77 | ~10 sec |

### What this teaches us

1. **More complex ≠ always better.** The Random Forest beats Logistic Regression by a solid margin. The ANN edges past Random Forest. But the CNN is barely different from the ANN.
2. **Diminishing returns.** Each step up in complexity gives a smaller gain. This is normal.
3. **Pick the simplest model that does the job.** If Logistic Regression gets you to acceptable accuracy and is interpretable, ship that. Deep learning is a great tool but it costs more time, memory, and explanation.

> **Rule of thumb:** start with the simplest model. Only add complexity if simpler models clearly fall short.

---

## Combine them all into one script

Try writing a single comparison script that trains all four models and prints a nice table. Skeleton:

```python
# compare_all.py
import os, pandas as pd, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten

# --- Load + preprocess (shared) ---
# (paste in your standard pipeline)

results = {}

# --- Logistic Regression ---
lr = LogisticRegression(solver='liblinear', random_state=42).fit(X_train_scaled, y_train)
results['LogReg'] = (
    accuracy_score(y_test, lr.predict(X_test_scaled)),
    roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train_scaled, y_train)
results['RandomForest'] = (
    accuracy_score(y_test, rf.predict(X_test_scaled)),
    roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
)

# --- ANN ---
ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(), Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
ann.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
ann.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
_, acc, auc = ann.evaluate(X_test_scaled, y_test, verbose=0)
results['ANN'] = (acc, auc)

# --- CNN ---
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)
cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    BatchNormalization(), Dropout(0.2),
    Conv1D(32, 3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
cnn.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)
_, acc, auc = cnn.evaluate(X_test_cnn, y_test, verbose=0)
results['CNN'] = (acc, auc)

# --- Print table ---
print(f"\n{'Model':<15}{'Accuracy':>10}{'AUC':>10}")
print('-' * 35)
for name, (a, u) in results.items():
    print(f"{name:<15}{a:>10.3f}{u:>10.3f}")
```

Try writing this yourself before peeking. It's a great exercise to consolidate everything.

---

## Back to the project's real question — fairness

We trained models that decide if a voice sounds Trustworthy. But the project is called **Echoes of Equity**: does the model work *equally well* for everyone?

Here is the experiment you can run yourself.

### Step 1 — Keep the demographic columns aside

Earlier we dropped `Speaker_Sex`, `Speaker_AgeGroup`, and `Speaker_Ethnicity` from `X`. Save them separately before dropping:

```python
demographics = df[['Speaker_Sex', 'Speaker_AgeGroup', 'Speaker_Ethnicity']].copy()
```

### Step 2 — After splitting, align them with the test set

`train_test_split` returns the same indices, so:

```python
X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
    X, y, demographics, test_size=0.2, stratify=y, random_state=42
)
```

### Step 3 — Score the model per group

```python
# Predict probabilities for the whole test set
y_prob = model.predict_proba(X_test_scaled)[:, 1]   # or model.predict(X_test_scaled).flatten() for Keras
y_pred = (y_prob >= 0.5).astype(int)

# Score per demographic group
import pandas as pd
test_results = demo_test.copy()
test_results['true']  = y_test.values
test_results['pred']  = y_pred
test_results['prob']  = y_prob

for group_col in ['Speaker_Sex', 'Speaker_AgeGroup', 'Speaker_Ethnicity']:
    print(f"\n--- {group_col} ---")
    for group, sub in test_results.groupby(group_col):
        if len(sub) < 10:
            continue   # too few samples to score reliably
        acc = (sub['true'] == sub['pred']).mean()
        try:
            auc = roc_auc_score(sub['true'], sub['prob'])
        except ValueError:
            auc = float('nan')  # happens if only one class is present in the group
        print(f"  {group:<20} n={len(sub):>3}  acc={acc:.2f}  auc={auc:.2f}")
```

### What to look for

If the model is fair, you should see similar accuracy and AUC across groups. **Gaps reveal bias** — and the bigger the gap, the more concerning.

If you find a gap, the next question is *why*. Some possibilities:

- The training data contained fewer samples from one group → less practice.
- Acoustic features themselves are different across groups (pitch differs by sex, formants differ by vocal-tract length) → the model leans on features that happen to correlate with demographics.
- The labels were collected by raters who themselves had biased intuitions about which voices "sound trustworthy".

This is the part where the *project* becomes the *paper*. Document the gaps, hypothesize causes, propose mitigations.

---

## What to learn next

You now know the core ML/DL toolbox. Reasonable next steps:

| Topic | Why | Where to learn |
|-------|-----|-----------------|
| **Cross-validation** | One train/test split is noisy — k-fold gives more reliable numbers | scikit-learn `cross_val_score` |
| **Hyperparameter tuning** | Random Forest's `n_estimators`, `max_depth`, ANN's layer sizes — try them systematically | `GridSearchCV`, `RandomizedSearchCV`, Optuna |
| **Pipelines** | Scaler + model in one object → no leakage, easier to reuse | scikit-learn `Pipeline` |
| **Confusion matrix** | Visualize which classes get confused | `sklearn.metrics.ConfusionMatrixDisplay` |
| **SHAP values** | Explain *why* the model predicted what it predicted, per sample | `shap` Python package |
| **Working with raw audio** | Mel-spectrograms + 2D CNN — the real next level for voice projects | `librosa` + Keras |

A great roadmap: pick one topic per week, apply it to *this* project. By the end of a couple of months you'll have a sophisticated, well-evaluated, interpretable ML system — and a deep understanding of how each piece works.

---

## Common bad habits to actively avoid

- **Comparing on training accuracy.** Useless. Always evaluate on held-out data.
- **Reporting only one number.** Always look at multiple metrics. Always look at per-group results.
- **Trusting one run.** Random seeds matter. Run multiple times with different seeds and report the mean and spread.
- **Confusing correlation with causation.** "Feature X has high importance" ≠ "X causes the label". It just means the model uses it.
- **Treating black-box models as oracles.** A 0.79 AUC model is *wrong about 1 in 5 cases*. Behave accordingly when the stakes are real.

---

## Closing thought

You started with no machine learning knowledge. You now have:

- Built four models that span the classical/deep, simple/complex spectrum.
- Learned the workflow: load → preprocess → split → scale → train → evaluate.
- Met the vocabulary (overfitting, regularization, gradients, activations) that every ML practitioner uses every day.
- The tools to ask *fairness* questions of your own models, not just accuracy ones.

That last point is the most important. Many people can train a model. Far fewer ask "for whom does it work?" Be one of the few.

---

## A final challenge

Pick **one** of these and do it this week:

1. Add SHAP explanations to the Random Forest model (`pip install shap`, then `explainer = shap.TreeExplainer(model)`).
2. Run the fairness analysis above on all four models and write a paragraph comparing them.
3. Build a 5th model: try `GradientBoostingClassifier` from sklearn and add it to your comparison table.
4. Save the trained ANN to a file (`model.save('my_ann.keras')`) and write a small script that loads it and classifies a single new row.

When you finish, you're not a beginner anymore.

---

Return to the **[README](README.md)**.
