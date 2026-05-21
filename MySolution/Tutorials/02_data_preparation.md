# Tutorial 02 — Data Preparation

In machine learning, **data preparation is 80% of the work**. Models can only learn from clean, numeric, well-scaled data. Garbage in → garbage out.

This tutorial walks through every preparation step you will see repeated in *all four* of our models. Learn it once, use it four times.

---

## The five-step preprocessing recipe

You will see this exact recipe in every script:

1. **Load** the CSV
2. **Encode** the label (turn text into a number)
3. **Drop** columns that shouldn't be features
4. **Handle** missing values
5. **Split** into train and test
6. **Scale** the numeric features

Let's walk through each step.

---

## Step 1 — Load the CSV

```python
import pandas as pd
import os

# Build the path the safe way: relative to where THIS script lives.
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')

df = pd.read_csv(csv_path)
print(df.shape)        # (rows, columns)
print(df.head())       # first 5 rows
```

### Why the `os.path` ceremony?

If you wrote just `pd.read_csv('Speech_dataset_characteristics.csv')`, the file would only be found if you happened to *run* the script from the right folder. The `os.path` version makes the path relative to **the script itself**, so it works no matter where you launch it from.

This is a small detail that will save you hours of debugging later.

---

## Step 2 — Encode the label

Our label column looks like this:

```
Neutral
Trustworthy
Neutral
Neutral
Trustworthy
...
```

Computers can't do math on the word "Neutral". We have to turn it into a number. The standard tool is `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])
```

After this line, the column contains 0s and 1s instead of words. By default, the encoder sorts alphabetically:

```
'Neutral'     → 0
'Trustworthy' → 1
```

> **Heads up:** `LabelEncoder` is great for the *target*. If you needed to encode an *input* feature with categories like "Red / Green / Blue", you'd use `OneHotEncoder` instead, because there is no natural ordering between colors. But for binary labels, `LabelEncoder` is perfect.

---

## Step 3 — Drop columns we shouldn't use as features

```python
X = df.drop(columns=[
    'Audio_Filename',      # just a filename — irrelevant
    'Speaker_ID',          # see "data leakage" warning below
    'Speaker_Ethnicity',   # demographic — we'll come back to this
    'Speaker_AgeGroup',    # demographic
    'Speaker_Sex',         # demographic
    'Speaker_Intent',      # this is the LABEL, not a feature!
    'Sentence_Num',        # which prompt they read — not acoustic
])
y = df['Speaker_Intent']
```

### Why drop the label from X?

If you include the label in your features, the model just learns "the answer is in column 6" and gets 100% accuracy on training while learning nothing real. That's a **data leak** — the test set's secrets have leaked into the training set.

### Why drop Speaker_ID?

This is the subtler kind of leak. Imagine the same speaker appears in both the training set and the test set. If the model can identify "this is speaker #1893 again, last time they were Neutral, so predict Neutral", it's cheating — it's memorizing the speaker, not the voice characteristics.

### Why drop the demographics?

For two reasons:

1. We want a model that judges based on **acoustic features**, not on whether the speaker is male/female or younger/older. (Otherwise we'd be baking bias straight in.)
2. After training, we will use these demographic columns to *evaluate fairness* — to ask "does the model work equally well for everyone?". They're held out as a measuring tool.

---

## Step 4 — Handle missing values

CSV files in the wild always have some holes:

- `NaN` (Not a Number) — pandas's symbol for "missing"
- `inf` / `-inf` — infinity (often from divisions by zero earlier in the pipeline)

We have two basic strategies:

| Strategy | Code | When to use |
|----------|------|-------------|
| Drop the row | `df.dropna()` | When missing rows are rare and you have lots of data |
| Fill ("impute") | `df.fillna(df.mean())` | When you don't want to lose any rows |

We will impute with the column mean — a safe, simple default:

```python
import numpy as np

# Replace infinities first (they would mess up the mean)
X = X.replace([np.inf, -np.inf], np.nan)

# Now fill NaNs with the column's mean
X = X.fillna(X.mean())
```

> **Subtle point:** Imputing with the mean of the *whole dataset* technically leaks a tiny bit of test-set info into the training set (the mean knows about test rows). The "proper" version computes the mean only on training data. For this small project it doesn't matter much, but in real research you'd use `SimpleImputer` from scikit-learn inside a pipeline.

---

## Step 5 — Train / test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% goes to the test set
    stratify=y,           # KEEP the Neutral/Trustworthy ratio the same in both sets
    random_state=42       # makes the random split reproducible
)
```

Three important arguments:

- **`test_size=0.2`** — 80/20 split. You'll also see 70/30 or 75/25; 80/20 is the most common default.
- **`stratify=y`** — *Critical.* Without this, you might randomly get a test set that's mostly Neutral and a training set that's mostly Trustworthy. Stratifying forces the same class proportions in both.
- **`random_state=42`** — Any integer. Makes your "random" split deterministic so you and your classmate get the same result. (Why 42? It's a joke from *The Hitchhiker's Guide to the Galaxy*.)

---

## Step 6 — Scale the features

This step is the one beginners most often forget.

### Why scaling matters

Look at two of our features:

| Feature | Typical values |
|---------|----------------|
| `Mean_Pitch(F0)` | 100 — 250 |
| `Local_Jitter` | 0.01 — 0.05 |

If a model is doing math with both columns at once, the pitch feature *dominates* simply because its numbers are bigger. Jitter, which might be the actually-important signal, gets drowned out.

**Scaling** rewrites every feature so they're all on the same playing field.

### `StandardScaler` — the workhorse

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean+std AND apply
X_test_scaled  = scaler.transform(X_test)        # only apply, don't re-fit
```

What this does to each column:

```
scaled_value = (original_value - column_mean) / column_std
```

After scaling, every column has mean ≈ 0 and standard deviation ≈ 1.

### The golden rule: `fit_transform` on train, `transform` only on test

> Never let the scaler "see" the test data when computing its mean and standard deviation.

If you call `fit_transform` on the test set, you're leaking test statistics into your preprocessing — another subtle form of data leakage. The scaler should learn from training data only, then apply that same learned transformation to the test data.

---

## Putting it all together

Here is the complete preprocessing block. **Type this into a new file and run it** to make sure it works on your machine:

```python
# preprocessing_demo.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Encode label
encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])
print(f"Label classes: {encoder.classes_}")  # ['Neutral' 'Trustworthy']

# 3. Drop non-feature columns
X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity',
                     'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']
print(f"Feature count: {X.shape[1]}")

# 4. Handle missing / infinite values
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 6. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"After scaling: train mean ≈ {X_train_scaled.mean():.4f}, std ≈ {X_train_scaled.std():.4f}")
```

Save this in `MySolution/ExistingMethods/` (next to the CSV), and run:

```bash
python preprocessing_demo.py
```

You should see something like:

```
Loaded 1191 rows, 65 columns
Label classes: ['Neutral' 'Trustworthy']
Feature count: 58
Train: (952, 58), Test: (239, 58)
After scaling: train mean ≈ -0.0000, std ≈ 1.0000
```

(Numbers will be close, not exact.)

---

## Check yourself

1. Why do we encode `'Neutral'` and `'Trustworthy'` as 0 and 1?
2. What is data leakage? Give one example.
3. Why does `train_test_split` use `stratify=y`?
4. What does `fit_transform` do that `transform` does not?
5. Why do we scale features before training?

If all five make sense, you're ready to train your first model in **[Tutorial 03 — Logistic Regression](03_logistic_regression.md)**.
