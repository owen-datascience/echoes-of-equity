### Lesson 5 – Supervised Learning & Baseline Models (Logistic Regression + Random Forest)

---

#### 1. Lesson Summary

In this lesson, the student will connect everything learned so far (features, labels, dataset structure, EDA) into a **complete supervised learning pipeline**.

We’ll focus on:

* What **supervised learning** is (using labeled data to learn patterns).
* What a **binary classifier** does (predict neutral vs trustworthy).
* How to:

  * Split data into **train** and **test** sets.
  * Train **Logistic Regression** and **Random Forest** models.
  * Evaluate them using **accuracy, precision, recall, and F1-score**.
* Understand these models as **baseline classifiers** that approximate what the paper did:

  * Logistic Regression: simple, linear, interpretable.
  * Random Forest: ensemble of decision trees capturing nonlinear patterns.

By the end, the student will know how to **train and evaluate simple models in Python** using scikit-learn on a trust-like acoustic feature dataset.

---

#### 2. Key Points

* **Supervised learning** uses input features (X) and known labels (y) to learn a mapping from X → y.
* Our problem is **binary classification**:

  * 0 = neutral intent
  * 1 = trustworthy intent
* The workflow:

  1. Load features and labels from a table.
  2. Split into **train** and **test** sets.
  3. Train a model on the train set.
  4. Evaluate on the test set using metrics.
* **Train set**: used for fitting the model.
* **Test set**: used only to evaluate how the model performs on unseen data.
* **Logistic Regression**:

  * Linear model that outputs probabilities.
  * Fast and easy to interpret (weights for each feature).
* **Random Forest**:

  * Many decision trees trained on random subsets.
  * Captures nonlinear relationships and interactions.
* Evaluation metrics:

  * **Accuracy** = (correct predictions / all predictions).
  * **Precision** = of all predicted “trustworthy,” how many were correct?
  * **Recall** = of all truly “trustworthy,” how many did we find?
  * **F1-score** = harmonic mean of precision and recall.
* These baselines help us check whether the features contain useful information and give a reference level to compare more complex models.

---

#### 3. Real-World Examples or Stories

1. **Spam vs non-spam email**
   Features: words used, sender, links → Label: spam (1) or not spam (0).
   The model learns from labeled examples and then predicts on new emails.

2. **Medical test for disease**
   Features: age, blood pressure, test results → Label: disease present or not.
   Precision and recall matter: we care about both false positives and false negatives.

3. **Credit card fraud detection**
   Features: transaction amount, location, time → Label: fraud or normal.
   We need a model that generalizes to new transactions it has never seen.

4. **Our project: trust in voices**
   Features: pitch, HNR, jitter, shimmer, etc. → Label: neutral vs trustworthy intent.
   The model must generalize to new speakers, not just memorize old ones.

---

#### 4. Terminology Explained

* **Supervised learning** – Training a model using labeled examples (X, y) so it can predict y for new X.
* **Classification** – Predicting discrete categories (0/1, A/B/C), not continuous numbers.
* **Binary classification** – Only two classes (e.g., neutral vs trustworthy).
* **Feature matrix (X)** – Table of input features (rows = examples, columns = features).
* **Label vector (y)** – One-dimensional array of target labels.
* **Train/test split** – Divide data into a training part and a testing part.
* **Overfitting** – When a model memorizes training data and performs badly on new data.
* **Generalization** – How well a model performs on unseen data.
* **Baseline model** – A simple model used as a reference point.
* **Logistic Regression** – A linear model that uses a logistic function to output probabilities for class 1.
* **Random Forest** – An ensemble of decision trees where each tree gets a random subset of data/features.

---

#### 5. How It Works – Supervised Pipeline for Trust Detection

Imagine we have the feature table:

```text
duration_seconds | mean_f0_hz | sd_f0_hz | hnr_db | shimmer_db | cpp_db | intent_label
--------------------------------------------------------------------------------------
1.65             |   190.3    |  22.1    |  11.0  |   0.34     |  15.2  | 1
1.78             |   175.2    |  18.3    |   9.6  |   0.43     |  13.1  | 0
...
```

**Step 1 – Split features and labels**

```python
X = data[["duration_seconds", "mean_f0_hz", "sd_f0_hz", "hnr_db", "shimmer_db", "cpp_db"]]
y = data["intent_label"]
```

**Step 2 – Train/test split**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)
```

* `test_size=0.3` → 30% for testing, 70% for training.
* `stratify=y` keeps class balance similar in train and test.

**Step 3 – Train Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
```

**Step 4 – Evaluate Logistic Regression**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)
```

These numbers tell us how good the model is on unseen data.

**Step 5 – Train Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

**Step 6 – Evaluate Random Forest**

Same metrics as above; compare to see which does better.

**Step 7 – Interpret as baselines**

* If both models are around, say, 0.7–0.8 accuracy on a realistic dataset, we know acoustic features contain real signal.
* In the paper, Random Forest and Logistic Regression achieve **above-chance performance** (better than 50%) when predicting trustworthy intent from acoustic features.
* We will later refine evaluation with cross-validation (particularly leave-one-speaker-out), but this basic split is a good training step.

---

#### 6. Practice Exercises

**Exercise 1 – Identify the Type of Problem**

We have acoustic features as numbers and want to predict `intent_label` (0 = neutral, 1 = trustworthy).

Is this:
a) Regression
b) Binary classification
c) Multiclass classification
d) Clustering

---

**Exercise 2 – Train/Test Split Reasoning**

Explain in 3–4 sentences why we should **not** train and test on the same data when building a model.

---

**Exercise 3 – Metric Matching**

Match each metric to its question:

1. Accuracy
2. Precision
3. Recall
4. F1-score

Questions:
a) Of all examples that are actually trustworthy, how many did we correctly predict as trustworthy?
b) Of all predictions we made as trustworthy, how many were correct?
c) What fraction of all predictions (neutral + trustworthy) did we get right?
d) How well do we balance precision and recall in a single number?

---

**Exercise 4 – Confusion Matrix Interpretation**

You have a binary confusion matrix:

|          | Predicted 0 | Predicted 1 |
| -------- | ----------- | ----------- |
| Actual 0 | 40          | 10          |
| Actual 1 | 5           | 25          |

a) How many total predictions were made?
b) How many true positives (TP)?
c) How many true negatives (TN)?
d) How many false positives (FP)?
e) How many false negatives (FN)?

---

**Exercise 5 – Comparing Models**

Suppose we train two models on the same data and get:

* Logistic Regression: Accuracy = 0.72, F1 = 0.70
* Random Forest: Accuracy = 0.78, F1 = 0.77

Which model seems better overall, and why?

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

Correct: **b) Binary classification**
We have two classes: neutral (0) and trustworthy (1).

---

**Exercise 2 – Sample Answer**

If we train and test on the same data, the model might simply memorize the training examples instead of learning general patterns. This would make it look very accurate on the training data, but it might fail badly on new data it has never seen. Using a separate test set lets us estimate how well the model will generalize in real-world situations.

---

**Exercise 3 – Answer**

1 → c) Accuracy – fraction of all predictions that were correct.
2 → b) Precision – of all predicted trustworthy, how many were truly trustworthy.
3 → a) Recall – of all truly trustworthy examples, how many we caught.
4 → d) F1-score – a single number combining precision and recall.

---

**Exercise 4 – Answer**

Total predictions = 40 + 10 + 5 + 25 = **80**.

* True Positives (TP, actual 1 & predicted 1) = **25**.
* True Negatives (TN, actual 0 & predicted 0) = **40**.
* False Positives (FP, actual 0 but predicted 1) = **10**.
* False Negatives (FN, actual 1 but predicted 0) = **5**.

---

**Exercise 5 – Sample Answer**

The Random Forest model seems better overall, because it has higher accuracy (0.78 vs 0.72) and higher F1-score (0.77 vs 0.70). This means it is making fewer mistakes overall and has a better balance of precision and recall than Logistic Regression on this dataset.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why use both Logistic Regression and Random Forest?
   **A:** They are different types of models. Logistic Regression is simple and linear; Random Forest is more flexible and nonlinear. Comparing them helps us see whether the relationship between features and trust is mostly linear or more complex.

2. **Q:** What does `max_iter=1000` do in Logistic Regression?
   **A:** It sets the maximum number of optimization steps the algorithm is allowed to take. Sometimes the default (like 100) is too low, so we increase it to ensure convergence.

3. **Q:** What is `stratify=y` in `train_test_split`?
   **A:** It keeps the class proportions (0s and 1s) similar in both the train and test sets, which is important for balanced evaluation.

4. **Q:** Do we always need 30% of data for testing?
   **A:** No, it’s a common choice but not a rule. Sometimes we use 20%, 10%, or cross-validation instead. The key is to have enough data to train and enough unseen data to evaluate.

5. **Q:** Why might Random Forest perform better than Logistic Regression?
   **A:** Because it can model nonlinear interactions between features and handle complex decision boundaries, whereas Logistic Regression only learns a single linear boundary.

6. **Q:** What happens if our model gets 0.5 accuracy?
   **A:** In a balanced binary problem, that’s about the same as random guessing. It suggests the model isn’t learning useful patterns from the features.

7. **Q:** Do we need to scale features for Random Forest?
   **A:** Not usually. Tree-based models are fairly scale-invariant. However, scaling can matter a lot for Logistic Regression and other linear models.

8. **Q:** What is a “baseline” exactly?
   **A:** A baseline is a simple starting model that we use to measure progress. If a more complex model doesn’t beat the baseline, it may not be worth the extra complexity.

9. **Q:** Can we look at feature importance in Random Forest?
   **A:** Yes. Random Forest provides feature importance values that show which features contribute most to splitting decisions. We’ll explore interpretation in a later lesson.

10. **Q:** Is one train/test split enough?
    **A:** It’s a start, but not ideal. In research (like the paper), we often use **cross-validation** (e.g., leave-one-speaker-out) to get more reliable estimates. That will come in a later lesson.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** What type of learning problem is neutral vs trustworthy intent?
a) Regression
b) Binary classification
c) Multiclass classification
d) Clustering

**Answer:** b) Binary classification.

---

**Q2.** What is the main purpose of the test set?
**Answer:** To evaluate how well the trained model generalizes to new, unseen data.

---

**Q3.** Which model outputs a probability for class 1 and then uses a threshold (usually 0.5) to decide the class?
**Answer:** Logistic Regression.

---

**Q4.** In Random Forest, each tree is trained on:
a) All data and all features
b) Random subsets of data and features
c) Only the test set
d) Only neutral utterances

**Answer:** b) Random subsets of data and features.

---

**Q5.** Accuracy = 0.8 means:
**Answer:** The model correctly classified 80% of all examples.

---

**Q6.** Which metric focuses on “Of all predicted positives, how many are correct?”
a) Accuracy
b) Precision
c) Recall
d) F1-score

**Answer:** b) Precision.

---

**Q7.** Which metric focuses on “Of all true positives, how many did we find?”
**Answer:** Recall.

---

**Q8.** What does F1-score combine?
**Answer:** It combines precision and recall into a single number (harmonic mean).

---

**Q9.** True or False: If a model has high training accuracy but low test accuracy, it may be overfitting.

**Answer:** True.

---

**Q10.** Why is logistic regression a good baseline model?
**Answer:** It is simple, fast, interpretable, and gives a clear starting point for performance.

---

#### 10. Mini Practice Project – Baseline Classifiers for Trust (with .zip)

**Mini Project Title:** *Training Logistic Regression & Random Forest on Synthetic Trust Data*

I’ve prepared a mini project folder with:

* `synthetic_trust_baseline_dataset.csv`

  * Columns:

    * `duration_seconds`
    * `mean_f0_hz`
    * `sd_f0_hz`
    * `hnr_db`
    * `shimmer_db`
    * `cpp_db`
    * `intent_label` (0 = neutral, 1 = trustworthy)

* `train_baseline_classifiers.py`

  * Every line is **fully commented** so a high school student can follow it.
  * Steps:

    * Load dataset via pandas
    * Split into train/test (`train_test_split`)
    * Train **Logistic Regression**
    * Train **Random Forest**
    * Compute accuracy, precision, recall, F1-score for each model
    * Print confusion matrices and classification reports
    * Provide hints for interpretation

* `README.txt`

  * Instructions on how to install dependencies and run the script.

**How the student should use it:**

1. Unzip the file.
2. Open a terminal in `lesson5_baseline_classifiers`.
3. Install packages (once):

   ```bash
   pip install pandas scikit-learn
   ```
4. Run:

   ```bash
   python train_baseline_classifiers.py
   ```
5. Compare:

   * Logistic Regression vs Random Forest accuracies.
   * Their precision, recall, and F1-scores.
   * How many false positives and false negatives each has.

**Mini Project Extension Ideas:**

* Change `test_size` (e.g., 0.2, 0.4) and see how results change.
* Try reducing feature set (only `mean_f0_hz` and `hnr_db`) and compare performance.
* Adjust Random Forest `n_estimators` (number of trees) and see if performance improves.

---

#### 11. References

* scikit-learn official tutorials:

  * “Supervised learning”
  * “Classification”
  * “Model evaluation: quantifying the quality of predictions”
* Beginner-friendly resources:

  * Blog posts or videos on:

    * “Logistic Regression explained simply”
    * “Random Forest explained visually”
    * “Precision, recall, and F1-score for beginners.”

---

#### 12. Additional Information / Teacher Tips

* **Connect back to the paper**
  Explain that the paper also uses **Logistic Regression and Random Forest** as main models, so the student is now doing what real researchers do—just on a smaller, synthetic dataset.

* **Encourage interpretation**
  After running the script, ask:

  * Which model did better?
  * Why might that be?
  * Which feature do you think is most important?

* **Prepare for next lessons**
  In later lessons, you can:

  * Introduce **cross-validation and leave-one-speaker-out**.
  * Use real feature tables from the paper.
  * Explore **feature importance** and model interpretation.
