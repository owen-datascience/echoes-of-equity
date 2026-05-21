# Echoes of Equity — Tutorial Series

**For high school students who know Python and Data Science, but have never built a machine learning or deep learning model.**

Welcome! By the end of this series, you will have built **four** working AI models from scratch that listen to the acoustic properties of a person's voice and predict whether they sound **Neutral** or **Trustworthy**. Even better, you will understand *why* every line of code is there.

---

## What you will build

| # | Model | Type | What it teaches |
|---|-------|------|-----------------|
| 1 | Logistic Regression | Classical ML | The simplest "classifier" — a straight-line decision boundary |
| 2 | Random Forest | Classical ML | An ensemble of decision trees that vote |
| 3 | Artificial Neural Network (ANN) | Deep Learning | A brain-like stack of layers |
| 4 | 1D Convolutional Neural Network (CNN) | Deep Learning | A pattern-scanner inspired by how we see |

---

## Tutorial order (please follow it!)

| Tutorial | Topic | Time |
|----------|-------|------|
| [00 — Introduction](00_introduction.md) | What problem are we solving? What is the dataset? | 15 min |
| [01 — Machine Learning Basics](01_machine_learning_basics.md) | Supervised learning, features, labels, training | 30 min |
| [02 — Data Preparation](02_data_preparation.md) | Encoding, scaling, train/test split | 30 min |
| [03 — Logistic Regression](03_logistic_regression.md) | Build your first classifier | 45 min |
| [04 — Random Forest](04_random_forest.md) | Build a tree-based ensemble | 45 min |
| [05 — Deep Learning Intro](05_deep_learning_intro.md) | Neurons, layers, activations, backpropagation | 45 min |
| [06 — Building an ANN](06_building_ann.md) | Build a neural network with TensorFlow | 60 min |
| [07 — Building a CNN](07_building_cnn.md) | Build a 1D convolutional network | 60 min |
| [08 — Comparing Models](08_comparing_models.md) | Which model wins? When? Where to go next | 30 min |

**Total: about 6 hours of focused work.** Spread it over a week — your brain needs time to digest.

---

## Setup checklist

Before you start, make sure you have these installed. Open a terminal and run:

```bash
pip install pandas numpy scikit-learn tensorflow
```

You will also need the dataset file `Speech_dataset_characteristics.csv`. It is already included in the project folders.

---

## How to use these tutorials

1. **Read first, then code.** Each tutorial explains the *why* before showing the *what*. Don't skip the explanations.
2. **Type the code yourself.** Copy-pasting feels fast but teaches almost nothing. Typing builds muscle memory.
3. **Break things on purpose.** Change a number, delete a line, see what happens. This is the fastest way to learn.
4. **Compare with the reference code.** The finished code lives in `../ExistingMethods/` and `../NewMethods/`. Use it as an answer key, not a starting point.

Ready? Open [00 — Introduction](00_introduction.md) and let's begin.
