# Tutorial 03 — Logistic Regression

Time to build your first model.

The reference implementation lives at `../ExistingMethods/LogisticRegression.py`. Don't open it yet. We'll build it from scratch, then compare.

---

## What is Logistic Regression?

Despite the name, it's a **classifier**, not a regressor.

The name is historical: the word "regression" here just means "fitting a curve". The curve in question is the **sigmoid** (S-shape), and it squashes any number into the range 0 to 1 — which we interpret as a *probability*.

```
  output ▲
       1 │           ____________
         │         ╱
     0.5 │       ╱
         │     ╱
       0 │___╱______________________▶  input
            -3   0   +3
```

### Plain-English summary

> Logistic Regression draws a single straight line (or in higher dimensions, a flat plane) that tries to separate the two classes. New samples on one side of the line are predicted as one class; samples on the other side as the other class.

For our problem: it draws a 58-dimensional plane through the cloud of voice samples, putting "Neutrals" on one side and "Trustworthys" on the other.

### Why start here?

- **Fast** — trains in milliseconds.
- **Interpretable** — every feature gets a *weight*. Big positive weight = "more of this feature → more likely Trustworthy". You can see exactly what the model learned.
- **A good baseline** — if a fancy model can't beat Logistic Regression, the fancy model isn't doing anything useful.

---

## What you will build

A script that:

1. Loads the speech CSV (you already know how from Tutorial 02).
2. Preprocesses it (you already know how!).
3. Trains a Logistic Regression model.
4. Prints accuracy, AUC, and a classification report.

---

## Step-by-step

### Setup

Create a new file. You can put it anywhere, but the easiest is right next to the CSV:

```
MySolution/ExistingMethods/my_logistic_regression.py
```

### Step 0 — Imports

```python
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
```

You've met all of these except `LogisticRegression`. That's the model class itself.

### Steps 1–6 — Preprocessing (copy from Tutorial 02)

```python
# 1. Load
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
data = pd.read_csv(csv_path)

# 2. Encode label
encoder = LabelEncoder()
data['Speaker_Intent'] = encoder.fit_transform(data['Speaker_Intent'])

# 3. Drop non-feature columns
X = data.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity',
                       'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = data['Speaker_Intent']

# 4. Handle infinities and NaNs
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Scale
scaler = StandardScaler()
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled  = scaler.transform(X_test)
```

### Step 7 — Create and train the model

```python
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train_rescaled, y_train)
```

That's it. Two lines.

#### Why `solver='liblinear'`?

A solver is the *algorithm* that figures out the best line. Logistic Regression has several solvers, optimized for different situations:

| Solver | Best for |
|--------|----------|
| `liblinear` | Small to medium datasets, binary classification (our case ✓) |
| `lbfgs` | The default; good for medium datasets, multi-class |
| `saga` | Very large datasets |

For our ~1,200 rows and 2 classes, `liblinear` is the cleanest choice.

#### Why `random_state=42`?

Most solvers involve some randomness at startup. Setting `random_state` makes the run reproducible.

### Step 8 — Predict

```python
predictions   = lr_model.predict(X_test_rescaled)            # returns 0s and 1s
probabilities = lr_model.predict_proba(X_test_rescaled)[:, 1] # probability of class 1
```

Two flavors of prediction:

- `predict` — returns a hard decision: 0 or 1.
- `predict_proba` — returns a probability for each class. We slice `[:, 1]` to keep only the probability of the *positive* class (Trustworthy). We need this for AUC.

### Step 9 — Evaluate

```python
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"Logistic Regression AUC:      {roc_auc_score(y_test, probabilities):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

Run your script:

```bash
python my_logistic_regression.py
```

You should see something like:

```
Logistic Regression Accuracy: 0.65
Logistic Regression AUC:      0.70

Classification Report:
              precision    recall  f1-score   support

           0       0.66      0.67      0.66       121
           1       0.65      0.64      0.64       118

    accuracy                           0.65       239
   macro avg       0.65      0.65      0.65       239
weighted avg       0.65      0.65      0.65       239
```

Your exact numbers may differ by a few percentage points depending on your Python/sklearn version.

---

## Reading the classification report

Each row is a class.

- Row `0` = Neutral, Row `1` = Trustworthy.
- **Precision** = "When I predict this class, how often am I right?"
- **Recall** = "Of all actual samples of this class, how many did I catch?"
- **F1-score** = the harmonic mean of precision and recall.
- **support** = how many test samples belonged to that class.

For a fair model, you want roughly equal precision and recall for both classes.

---

## Bonus — look at what it learned

One of the joys of Logistic Regression is interpretability. Add this to the end of your script:

```python
# Show which features most influenced the model
feature_names = X.columns
coefficients  = lr_model.coef_[0]   # one weight per feature

# Sort by absolute importance and display the top 10
importance = pd.DataFrame({'feature': feature_names, 'weight': coefficients})
importance['abs_weight'] = importance['weight'].abs()
print("\nTop 10 most influential features:")
print(importance.sort_values('abs_weight', ascending=False).head(10).to_string(index=False))
```

You will see acoustic features ranked by how much they pushed the prediction toward Trustworthy (positive weight) or Neutral (negative weight).

Spend a minute looking at the top features. Are they pitch-related? Jitter? Formants? This is your first taste of **model interpretation** — figuring out *why* a model makes its decisions.

---

## Common mistakes to watch for

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgot to scale | Accuracy near 0.5 or wild numbers in coefficients | Always `StandardScaler` |
| Used `X.values` instead of `X` before fit | Loses feature names for the bonus step | Use `X` (a DataFrame) |
| Called `fit_transform` on test set | Test results look better than they should | Test must use `transform` only |
| Forgot `stratify=y` | Class imbalance differs train vs test | Always stratify for classification |

---

## Compare your code

Now open `../ExistingMethods/LogisticRegression.py`. Read it line by line and compare to yours. Differences are fine — there is more than one right way — but they should all reach the same final accuracy and AUC.

---

## Check yourself

1. What does the sigmoid function do?
2. Why is Logistic Regression a good first model to try?
3. What is the difference between `predict` and `predict_proba`?
4. Where in the classification report would you look to ask "is the model fair to both classes?"

When those click, head to **[Tutorial 04 — Random Forest](04_random_forest.md)** for a much more flexible model.
