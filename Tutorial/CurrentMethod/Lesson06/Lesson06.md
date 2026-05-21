### Lesson 6 – Cross-Validation & Leave-One-Speaker-Out Evaluation

---

#### 1. Lesson Summary

In this lesson, the student learns **how to evaluate trust-detection models in a fair, research-grade way**, especially for **voice data**.

We’ll focus on:

* What **cross-validation** is and why it’s better than a single train/test split.
* Why voice models must be evaluated in a **speaker-independent** way.
* How **Leave-One-Speaker-Out (LOSO) cross-validation** works:

  * Train on all speakers except one.
  * Test on the held-out speaker.
  * Repeat for every speaker, then average the results.
* How to implement LOSO using **scikit-learn’s `LeaveOneGroupOut`**.

This mirrors what the research paper does when evaluating models on the trustworthy-intent dataset: they want to know if a model can predict trust **for new voices, not just the ones it has already heard**.

---

#### 2. Key Points

* A **single train/test split** is useful but can be unstable or lucky/unlucky depending on the split.
* **Cross-validation** (CV) reduces this risk by using multiple splits and averaging performance.
* For voice data, **speaker leakage** (same speaker in train & test) can give **over-optimistic results**.
* **Speaker-independent evaluation** means:

  * Speakers in the test set are **never** in the train set.
* **Leave-One-Speaker-Out (LOSO)**:

  * Each fold: hold out one speaker for testing.
  * Train on all other speakers.
  * Repeat for every speaker, average metrics.
* LOSO is stricter and more realistic for tasks like “can we detect trust in the voice of a **new** person?”
* scikit-learn’s `LeaveOneGroupOut` makes LOSO easy when you have a `speaker_id` column.
* Metrics (accuracy, precision, recall, F1) are computed **per fold** and then averaged.

---

#### 3. Real-World Examples or Stories

1. **Teacher grading new students**
   If you design a test and give practice questions, you want to know: “Will this test work for next year’s students who weren’t in my practice group?” LOSO mimics that: we test on **one new student** at a time.

2. **Voice assistant for new users**
   A company building a voice assistant must ensure it works for **people it never heard before**. If they test on the same people used for training, performance will look unrealistically high.

3. **Coach picking a team strategy**
   A coach tries strategies across different games, not just in one scrimmage. Cross-validation is like trying your strategy across many “games” (folds) to see if it’s robust.

4. **Language learning app**
   An app that recognizes pronunciation should work for new learners, not just the ones it was trained on. LOSO is like holding out one learner at a time.

---

#### 4. Terminology Explained

* **Cross-validation (CV)** – Evaluating a model by training & testing multiple times on different splits and averaging results.
* **k-fold cross-validation** – Split the data into k equal parts; each part is used as test once, and we average across k runs.
* **Groups** – A column (like `speaker_id`) that tells us which rows belong together.
* **Speaker leakage** – When recordings from the same speaker appear in both training and test sets; the model may “recognize the speaker” instead of learning general patterns.
* **Speaker-independent evaluation** – Evaluation where all speakers in the test set are different from those in the training set.
* **Leave-One-Group-Out (LOGO / LOSO)** – A form of cross-validation where each group (here, each speaker) is left out once as the test set.
* **Fold** – One round of train/test split in cross-validation.
* **Mean ± standard deviation of metrics** – Average performance and how much it varies from fold to fold.

---

#### 5. How It Works – LOSO in Practice

We have a dataset with columns like:

```text
speaker_id | duration_seconds | mean_f0_hz | sd_f0_hz | hnr_db | shimmer_db | cpp_db | intent_label
--------------------------------------------------------------------------------------------------
S01        | 1.68             | 180.2      | 18.5     | 9.6    | 0.42       | 13.1   | 0
S01        | 1.61             | 192.3      | 22.0     | 11.2   | 0.35       | 15.0   | 1
...
S16        | ...              | ...        | ...      | ...    | ...        | ...    | ...
```

We want to test: **If we train on 15 speakers, can we predict trust for the 16th (new) speaker?**

**Step 1 – Prepare X, y, and groups**

```python
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]
X = data[feature_columns]          # features
y = data["intent_label"]           # labels
groups = data["speaker_id"]        # speaker IDs
```

**Step 2 – Set up LeaveOneGroupOut**

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
```

**Step 3 – Loop over folds**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

accuracies = []
precisions = []
recalls = []
f1_scores = []

fold_index = 0
for train_idx, test_idx in logo.split(X, y, groups):
    fold_index += 1

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    test_speakers = groups.iloc[test_idx].unique()
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    
    print(f"Fold {fold_index} - Test speaker(s): {list(test_speakers)}")
    print("  Accuracy:", acc)
    print("  Precision:", prec)
    print("  Recall:", rec)
    print("  F1-score:", f1)
    print()
```

**Step 4 – Average metrics**

```python
accuracies = np.array(accuracies)
print("Mean accuracy:", accuracies.mean())
print("Std accuracy:", accuracies.std())
# Similarly for precision, recall, F1
```

**Interpretation:**

* Each fold answers: *“How well does the model work for this new speaker?”*
* The average across folds answers: *“On average, how well do we handle new speakers?”*
* The standard deviation shows how much performance changes from speaker to speaker.

This is **exactly the kind of speaker-independent evaluation** we want for the trustworthy-intent dataset.

---

#### 6. Practice Exercises

**Exercise 1 – Why Speaker-Independent?**

Explain in 2–3 sentences why testing on the **same speakers** you trained on can give you an overly optimistic view of model performance in a voice trust-detection task.

---

**Exercise 2 – Comparing Evaluation Methods**

List one advantage and one disadvantage of:

a) Single random train/test split
b) k-fold cross-validation
c) Leave-One-Speaker-Out (LOSO)

---

**Exercise 3 – Folds in LOSO**

Suppose you have 12 speakers in your dataset.

a) How many folds will LOSO create?
b) In each fold, roughly what fraction of the data is used for training vs testing?

---

**Exercise 4 – Metric Averaging**

You run LOSO and get accuracy scores per fold:

```text
[0.75, 0.80, 0.70, 0.85, 0.65, 0.90]
```

a) What is the mean accuracy?
b) (Conceptual) What does a larger standard deviation tell you about performance across speakers?

---

**Exercise 5 – Implementing Groups**

In LOSO, we use `groups = data["speaker_id"]`.

a) What would happen if we mistakenly used `groups = data["intent_label"]`?
b) Why is that wrong for speaker-independent evaluation?

---

#### 7. Solutions / Model Answers

**Exercise 1 – Sample Answer**

If we test on the same speakers we trained on, the model can learn **speaker-specific quirks** (like a person’s usual pitch) instead of general patterns of trustworthy vs neutral speech. That makes results look better than they really are—when we apply the model to a new person, performance may drop. Speaker-independent testing avoids this by always evaluating on new speakers.

---

**Exercise 2 – Sample Answer**

a) **Single split**

* Advantage: Simple and fast.
* Disadvantage: Results can be unstable and depend on which rows ended up in train vs test.

b) **k-fold cross-validation**

* Advantage: More reliable estimates by averaging over k different splits.
* Disadvantage: More computation; if we split purely by rows, we might still mix speakers between train and test.

c) **LOSO**

* Advantage: Strong, realistic test of generalization to new speakers.
* Disadvantage: Can be slower (one fold per speaker) and needs enough data per speaker.

---

**Exercise 3 – Answer**

a) LOSO will create **12 folds**, one per speaker.
b) In each fold, we train on **11/12 of the data** and test on **1/12 of the data** (the held-out speaker’s utterances).

---

**Exercise 4 – Answer**

a) Mean accuracy:
[
(0.75 + 0.80 + 0.70 + 0.85 + 0.65 + 0.90) / 6 = 4.65 / 6 \approx 0.775
]
So average accuracy ≈ **0.78** (77.5%).

b) A larger standard deviation means performance varies a lot between speakers: the model might work very well for some speakers and poorly for others, which is important to know.

---

**Exercise 5 – Sample Answer**

a) If we set `groups = data["intent_label"]`, LOSO would hold out **all neutral utterances** in one fold and all trustworthy utterances in another fold, instead of holding out one speaker at a time.

b) That is wrong because we want to test on new **speakers**, not on new labels. Using labels as groups breaks the idea of speaker-independence and makes the evaluation meaningless for the “new speaker” scenario.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why not always use LOSO instead of k-fold?
   **A:** LOSO is great when you have a clear grouping like speakers. For other tasks (like images of cats and dogs), there may be no natural group, so regular k-fold is more appropriate.

2. **Q:** Is LOSO always better than a single split?
   **A:** Yes, for grouped data like this, LOSO gives a more reliable and realistic estimate, though it’s more computationally expensive.

3. **Q:** Do we need to shuffle data when using LOSO?
   **A:** LOSO uses groups, so shuffling rows isn’t as important for the split itself. But shuffling can still help when training some models internally.

4. **Q:** How many utterances per speaker do we need?
   **A:** More is better. With very few utterances per speaker, each test fold might be too small to get stable metrics.

5. **Q:** Can we use Random Forest with LOSO?
   **A:** Yes. Any classifier (Logistic Regression, Random Forest, etc.) can be used in the LOSO loop; you just fit it inside the fold loop.

6. **Q:** Why do we average metrics across folds?
   **A:** Each fold gives performance for one test speaker. Averaging gives an overall picture of how the model performs across all speakers.

7. **Q:** What if one speaker is much harder to predict than others?
   **A:** That would show up as a low accuracy for that fold and a higher standard deviation. We might investigate what’s special about that speaker.

8. **Q:** Should we tune hyperparameters inside LOSO?
   **A:** In proper research, yes: you’d nest another cross-validation inside the training set for hyperparameter tuning. For learning purposes, we usually start with default settings.

9. **Q:** Does LOSO guarantee no overfitting?
   **A:** No model can fully escape overfitting, but LOSO greatly reduces over-optimistic estimates due to speaker leakage.

10. **Q:** How does this connect to the trust paper?
    **A:** The paper evaluates models in a **speaker-independent** manner, similar to LOSO, to demonstrate that acoustic cues of trustworthy intent generalize to new voices, not just the ones in training.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** What is the main goal of cross-validation?
**Answer:** To get a more reliable estimate of a model’s performance by training and testing on multiple splits of the data and averaging the results.

---

**Q2.** In LOSO, what is left out in each fold?
a) One random utterance
b) One random feature
c) One speaker (group)
d) One class

**Answer:** c) One speaker (group).

---

**Q3.** Why is speaker leakage a problem?
**Answer:** Because the model can learn to recognize individual speakers instead of general patterns, giving overly optimistic test performance that won’t hold for new speakers.

---

**Q4.** If you have 20 speakers, how many folds does LOSO create?
**Answer:** 20 folds.

---

**Q5.** True or False: In LOSO, each utterance is used in the test set exactly once.

**Answer:** True.

---

**Q6.** Which scikit-learn class is used for LOSO?
a) `KFold`
b) `StratifiedKFold`
c) `LeaveOneOut`
d) `LeaveOneGroupOut`

**Answer:** d) `LeaveOneGroupOut`.

---

**Q7.** What does the `groups` argument represent in `logo.split(X, y, groups)`?
**Answer:** It represents the grouping variable (here, `speaker_id`) that tells which rows belong to each speaker.

---

**Q8.** If average accuracy is 0.80 and standard deviation is 0.02, what does that mean?
**Answer:** The model’s accuracy is about 80% on average, and performance is fairly consistent across speakers (only about ±2% variation).

---

**Q9.** If one speaker’s fold has 0.50 accuracy while others are around 0.80, what might you do?
**Answer:** Investigate that speaker: maybe their voice is very different, or there was a data issue. It shows the model struggles more for that speaker.

---

**Q10.** Why is LOSO especially important for the trust-in-voice dataset?
**Answer:** Because we care about detecting trust for **new voices**, not just the ones we trained on. LOSO directly tests this ability.

---

#### 10. Mini Practice Project – Leave-One-Speaker-Out Cross-Validation (Downloadable .zip)

**Mini Project Title:** *LOSO Cross-Validation for Trust from Acoustic Features*

I’ve created a mini project folder that includes:

* `synthetic_trust_loso_dataset.csv`

  * Columns:

    * `speaker_id` – e.g., S01, S02, …
    * `duration_seconds`
    * `mean_f0_hz`
    * `sd_f0_hz`
    * `hnr_db`
    * `shimmer_db`
    * `cpp_db`
    * `intent_label` (0 = neutral, 1 = trustworthy)

* `loso_cross_validation.py`

  * Every line is commented to explain what it does.
  * Steps:

    * Load dataset with pandas
    * Build `X`, `y`, and `groups = speaker_id`
    * Use `LeaveOneGroupOut` to create LOSO folds
    * Train Logistic Regression on each fold
    * Compute accuracy, precision, recall, F1 for each test speaker
    * Print per-speaker metrics and overall mean ± std

* `README.txt` with clear run instructions.

**How the student should use it:**

1. Unzip the file.
2. Open a terminal / command prompt in `lesson6_leave_one_speaker_out`.
3. Install packages if needed:

   ```bash
   pip install pandas scikit-learn numpy
   ```
4. Run:

   ```bash
   python loso_cross_validation.py
   ```
5. Look at:

   * Per-fold metrics (one fold per speaker).
   * Average accuracy, precision, recall, F1.
   * Standard deviations.
6. Reflect:

   * Does the model generalize similarly well to all speakers?
   * Which speaker fold is hardest? Why might that be?

---

#### 11. References

* scikit-learn documentation:

  * “Cross-validation: evaluating estimator performance” (section on `LeaveOneGroupOut`).
* Introductory articles or videos on:

  * “k-fold cross-validation”
  * “Group cross-validation in scikit-learn”
  * “Speaker-independent evaluation for speech models”

---

#### 12. Additional Information / Teacher Tips

* **Connect to research**
  Point out that LOSO or similar speaker-independent protocols are standard in speech research. The student is now doing **research-style evaluation**, not just toy splits.

* **Encourage experiments**
  Have the student:

  * Swap Logistic Regression for Random Forest in the script.
  * Compare average metrics and discuss why results change.

* **Lead into next lessons**
  Next steps could include:

  * Feature importance analysis (which acoustic cues matter most).
  * Model interpretation (e.g., logistic regression coefficients, SHAP later).
