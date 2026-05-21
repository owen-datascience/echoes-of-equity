# Tutorial 01 — Machine Learning Basics

You already know how to write a function in Python:

```python
def is_even(n):
    return n % 2 == 0
```

You told the computer **exactly** what rule to apply. That's traditional programming.

**Machine learning flips this.** Instead of writing the rule, you give the computer a bunch of examples and let it figure out the rule on its own.

```
Traditional:    Data + Rules  →  Answers
Machine Learn:  Data + Answers → Rules
```

That last "rule" is what we call a **model**.

---

## The three flavors of machine learning

| Flavor | What you give | What you get | Example |
|--------|---------------|--------------|---------|
| **Supervised** | Examples with correct answers | A model that predicts the answer for new examples | Trustworthy vs Neutral voice |
| **Unsupervised** | Examples *without* answers | Groupings / structure in the data | "These 100 songs cluster into 3 styles" |
| **Reinforcement** | A game / environment + a reward | An agent that learns to maximize the reward | A bot that learns to play chess |

**Everything in this project is supervised learning.** Forget the other two for now.

---

## The vocabulary you must know

Imagine a giant spreadsheet:

|  | Mean_Pitch | Jitter | Shimmer | ... | Speaker_Intent |
|---|-----------|--------|---------|-----|----------------|
| row 1 | 173.9 | 0.028 | 0.090 | ... | Neutral |
| row 2 | 163.4 | 0.013 | 0.095 | ... | Neutral |
| row 3 | 220.7 | 0.041 | 0.122 | ... | Trustworthy |

- **Feature** (`X`) — one of the input columns. We have ~50 of these.
- **Label** or **target** (`y`) — the column we want to predict. We have one: `Speaker_Intent`.
- **Sample** / **instance** — one row.
- **Dataset** — the whole spreadsheet.

By convention in Python machine learning code:

```python
X = features    # capital X, because it's a matrix (many rows × many columns)
y = labels      # lowercase y, because it's a vector (one value per row)
```

---

## Classification vs Regression

Supervised learning splits into two:

| If the label is… | The task is called… | Example |
|------------------|---------------------|---------|
| A category (one of a few options) | **Classification** | Neutral vs Trustworthy |
| A number (could be anything) | **Regression** | Predict house price |

We are doing **binary classification**: two categories. That is the simplest kind.

---

## The most important habit: train / test split

> "How do I know if my model actually learned anything, instead of just memorizing the data?"

This is the whole reason we split our data into two pieces:

```
                   ┌────────────────────────┐
   All data  ───▶  │   80% training set     │  ───▶  Model learns from this
                   ├────────────────────────┤
                   │   20% test set         │  ───▶  Model has NEVER seen this.
                   └────────────────────────┘         Used to grade the model.
```

If a student takes a practice test the night before and aces it, that's expected. If they ace the *real* test (questions they haven't seen), that's actual learning.

The same logic applies. We always evaluate models on data they did not train on.

---

## The "fit" / "predict" pattern

Almost every model in scikit-learn (the most popular Python ML library) follows the same recipe:

```python
model = SomeKindOfModel()       # 1. Create an empty model
model.fit(X_train, y_train)     # 2. Train it ("learn from the answers")
predictions = model.predict(X_test)   # 3. Use it on new data
```

That's it. Learn this pattern and you have learned 80% of scikit-learn.

---

## How do we know if the model is any good?

You can't just ask "is it right?" because that's not a single number. We use **metrics**:

### Accuracy

```
Accuracy = (number of correct predictions) / (total predictions)
```

- 1.0 = perfect, 0.5 = no better than flipping a coin (for two classes), 0.0 = always wrong.
- Easy to understand, but can lie if one class is much more common than the other.

### AUC (Area Under the ROC Curve)

This one is harder to explain in words. Think of it this way:

> AUC asks: "If I randomly pick one Neutral sample and one Trustworthy sample, how often does the model give a higher trustworthy-score to the actually-Trustworthy one?"

- 1.0 = always — perfect ranking
- 0.5 = random
- < 0.5 = somehow worse than random (you're holding your model upside down)

AUC is less fooled by imbalanced datasets and is often more informative than accuracy. We will report both.

### Classification report

Scikit-learn can also print a per-class summary with:

- **Precision** — Of the times the model predicted "Trustworthy", how often was it right?
- **Recall** — Of all actually-Trustworthy samples, how many did the model catch?
- **F1-score** — A balance between precision and recall.

You don't need to memorize these formulas now. Just know that `classification_report` shows them.

---

## Overfitting — the #1 thing to watch for

**Overfitting** happens when your model learns the training data *too well*, including the random noise, and then fails on new data.

```
                           Training score
                                ▲
                          ╔═════╪═════╗
                          ║ OVER-FIT ║   ← good on train, bad on test
                          ╚═════╪═════╝
                                │
                          ┌─────┴─────┐
                          │ JUST RIGHT│   ← good on both
                          └─────┬─────┘
                                │
                          ╔═════╪═════╗
                          ║ UNDER-FIT ║   ← bad on both (model is too simple)
                          ╚═══════════╝
                                ▼
                            Test score
```

Most of the choices you'll see (`max_depth=10`, `Dropout(0.3)`, etc.) are tricks to prevent overfitting. Tutorial 03 onward will call these out as we meet them.

---

## Check yourself

1. What is the difference between a feature and a label?
2. Why do we hold out a test set?
3. If a model gets 95% on training data and 55% on the test set, what is happening?
4. What does AUC = 1.0 mean? What does AUC = 0.5 mean?

If those click for you, head to **[Tutorial 02 — Data Preparation](02_data_preparation.md)**.
