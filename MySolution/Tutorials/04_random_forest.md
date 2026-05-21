# Tutorial 04 — Random Forest

In Tutorial 03 you trained a model that draws a single straight line through the data. Now we'll graduate to something much more flexible: a **forest of decision trees** that vote on the answer.

Reference implementation: `../ExistingMethods/RandomForest.py`.

---

## Idea 1 — The decision tree

Before there can be a forest, there must be a tree.

A decision tree is a sequence of yes/no questions about the features. Like a flowchart:

```
                   ┌────────────────────────────┐
                   │ Is Mean_Pitch > 180?       │
                   └─────────────┬──────────────┘
                          yes  ──┴── no
                       │                  │
              ┌────────▼─────────┐   ┌─────▼────────────┐
              │ Is Jitter > 0.02?│   │ Is HNR > 12?     │
              └────────┬─────────┘   └─────┬────────────┘
                yes  ──┴── no       yes  ──┴── no
                │           │       │            │
            Trustworthy   Neutral   Neutral    Trustworthy
```

The tree learns those questions automatically by looking at the training data and asking "which split separates Neutral from Trustworthy the best at this point?"

**Strengths:**
- Easy to interpret — you can literally read the tree.
- Handles non-linear patterns (Logistic Regression cannot).
- No scaling required (but it doesn't hurt to scale anyway).

**Weakness:**
- A single tree can be unstable: small changes in the data lead to very different trees. They tend to *overfit* — memorizing the training data.

---

## Idea 2 — From one tree to a forest

The fix is delightfully simple: **train hundreds of trees**, each on a slightly different random sample of the data, and let them **vote**.

```
       Sample 1 ──▶ Tree 1 ──▶ "Trustworthy"
       Sample 2 ──▶ Tree 2 ──▶ "Neutral"
       Sample 3 ──▶ Tree 3 ──▶ "Trustworthy"
          ...                       ...
      Sample 100 ──▶ Tree 100 ─▶ "Trustworthy"
                                    │
                                    ▼
                             Majority vote: Trustworthy
```

This is called a **Random Forest**, and it's one of the most reliable workhorses in classical ML. Each individual tree might overfit, but their *average* doesn't.

This idea — averaging many weak models to get one strong model — is called **ensembling**, and Random Forests are an example of a specific ensembling technique called **bagging**.

---

## What you will build

The same preprocessing as Tutorial 03, but the model class changes from `LogisticRegression` to `RandomForestClassifier`. That's it.

---

## Step-by-step

Create a new file in the same folder: `my_random_forest.py`.

### Imports — almost identical to last time

```python
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier          # ← changed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
```

Only difference: `RandomForestClassifier` instead of `LogisticRegression`. It lives in `sklearn.ensemble`, hinting at its ensemble nature.

### Preprocessing — identical to Tutorial 03

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

le = LabelEncoder()
df['Speaker_Intent'] = le.fit_transform(df['Speaker_Intent'])

features = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity',
                            'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
target = df['Speaker_Intent']

features = features.fillna(features.mean())

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Train the forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100,    # how many trees
    max_depth=10,        # how deep each tree can grow
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
```

Two new hyperparameters worth understanding:

#### `n_estimators=100` — number of trees

> "How many votes are we collecting?"

- More trees → more stable predictions, but slower to train and predict.
- 100 is a very common starting value. Bumping it to 500 or 1000 sometimes helps a bit but rarely a lot.

#### `max_depth=10` — how deep each tree grows

> "How many yes/no questions in a row before we stop?"

- Deeper trees can learn more nuanced patterns but overfit easily.
- Shallow trees underfit.
- `max_depth=10` is a sane middle ground for our small dataset.

These are called **hyperparameters** — settings *you* pick before training, as opposed to the parameters (the splits inside each tree) which the model *learns*. We will talk about tuning hyperparameters in Tutorial 08.

### Evaluate

```python
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Random Forest AUC:      {roc_auc_score(y_test, y_prob):.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

Run it. You should see something like:

```
Random Forest Accuracy: 0.70
Random Forest AUC:      0.75

Detailed Classification Report:
              precision    recall  f1-score   support
           0       0.71      0.68      0.69       121
           1       0.69      0.72      0.70       118
    accuracy                           0.70       239
   macro avg       0.70      0.70      0.70       239
weighted avg       0.70      0.70      0.70       239
```

Notice the model is *better* than Logistic Regression for the same data. That's the value of moving from a single linear model to an ensemble of nonlinear trees.

---

## Bonus — feature importances

Random Forests automatically compute how important each feature was. Add this:

```python
feature_names = features.columns
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))
```

Random Forest importance is calculated differently than Logistic Regression's weights:

- **Logistic weights** → direction *and* magnitude (positive pushes toward class 1).
- **Random Forest importances** → only magnitude (how useful the feature is overall), always positive, sum to 1.

Compare your top 10 with the top 10 from Tutorial 03. Do they overlap? Sometimes they agree, sometimes they don't — different model families "see" the data differently.

---

## Where does the randomness come from?

A Random Forest is "random" in two ways:

1. **Bootstrap sampling** — each tree is trained on a random sample of rows (drawn *with replacement*). Some rows appear multiple times; others get left out.
2. **Random feature subsets** — at each split, the tree only considers a random subset of the features. This forces different trees to focus on different aspects of the data.

Together, these two tricks make the trees *diverse*, which is what makes the ensemble strong.

---

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `n_estimators=1` | Highly unstable predictions | Use ≥ 100 |
| `max_depth=None` on small data | Severe overfitting | Set a cap (5–15 is typical) |
| Forgot `random_state` | Different result every run | Set `random_state=42` |
| Tried to interpret a single tree | Single trees are noisy | Use `feature_importances_` instead |

---

## Self-check

1. Why is a forest better than a single tree?
2. What is bagging?
3. What is the difference between a parameter and a hyperparameter?
4. Why does a Random Forest not need feature scaling? (We still did it for consistency.)

When ready, head to **[Tutorial 05 — Deep Learning Intro](05_deep_learning_intro.md)**, where we leave classical ML behind and meet neural networks.
