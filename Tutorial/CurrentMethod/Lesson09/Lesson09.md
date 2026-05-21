### Lesson 9 – End-to-End Speaker-Independent Trust Classification Pipeline

---

#### 1. Lesson Summary

This lesson pulls everything together into a **single, end-to-end pipeline** that looks a lot like what the research paper is doing:

1. We start from an utterance-level dataset that already has:

   * **Speaker IDs**
   * **Acoustic features**
   * **Aggregated human trust ratings** (mean, std, count)
   * **Binary trust label** (`intent_label`)

2. We then:

   * Build **feature matrix (X)**, **labels (y)**, and **groups (speaker_id)**.
   * Run **Leave-One-Speaker-Out (LOSO)** cross-validation.
   * Train **two models**: Logistic Regression and Random Forest.
   * Compute **accuracy, precision, recall, and F1** for each held-out speaker.
   * Average metrics across speakers and compare models.

By the end of this lesson, the student will understand how to combine **human labels + acoustic features + LOSO + models + metrics** into a coherent, research-style evaluation pipeline for trust in voices.

---

#### 2. Key Points

* An **end-to-end pipeline** connects all steps from data to evaluation:
  ratings → labels → features → splitting → training → metrics.
* **Speaker-independent evaluation** (LOSO) is baked into the pipeline via `LeaveOneGroupOut`.
* We compare **at least two models** (Logistic Regression vs Random Forest) to see:

  * Which performs better.
  * How model complexity affects generalization.
* We look at **average performance** across all speakers and **variation** (std).
* This is the bridge between **“toy scripts”** and **reproducing a real paper’s results**.
* Once this pipeline is clear, swapping in the **real dataset and code from the paper** is mostly an engineering task, not a conceptual one.

---

#### 3. Real-World Examples or Stories

1. **Tournament scoring**
   A sports league doesn’t decide the champion from one game; it tracks scores across all matches and all teams. Our pipeline is like the league table: it summarizes performance across all speakers, not just one convenient split.

2. **Manufacturing QA**
   A factory pipeline checks raw materials, assembly, and final inspection. Each step must work. Similarly, our ML pipeline needs: clean labels, good features, fair evaluation, and clear metrics.

3. **College admissions**
   Colleges look at transcripts, test scores, essays, recommendations, etc. This is an end-to-end process. Our pipeline similarly looks at multiple components to judge how good a model really is.

4. **Voice-based security system**
   A company building a trust-sensitive voice system must test it on many speakers, not just a few. They’d build almost exactly this kind of LOSO pipeline internally.

---

#### 4. Terminology Explained

* **End-to-end pipeline** – A full workflow from raw-ish data (features + labels) through to evaluation metrics.
* **Model family** – Type of model (e.g., Logistic Regression, Random Forest).
* **Speaker-independent** – Train on some speakers, test on different, unseen speakers.
* **Fold-level metrics** – Metrics computed on one train/test split (here, for one test speaker).
* **Aggregate metrics** – Mean and standard deviation of metrics across folds.
* **Baseline vs stronger model** – Logistic Regression is a simple baseline; Random Forest is more flexible and often stronger.

---

#### 5. How It Works – Building the Full Pipeline

We use a synthetic dataset that has one row per **utterance**:

```text
utterance_id | speaker_id | duration_seconds | mean_f0_hz | sd_f0_hz | hnr_db | shimmer_db | cpp_db | mean_rating | std_rating | n_ratings | intent_label
-------------------------------------------------------------------------------------------------------------------------------------------
U001         | S01        | 1.63             | 188.4      | 21.7     | 10.9   | 0.38       | 14.9   | 4.83        | 0.61       | 12        | 1
...
```

**Step 1 – Load and inspect**

```python
import pandas as pd

data = pd.read_csv("synthetic_trust_full_pipeline_dataset.csv")
print(data.head(8))
print("Shape:", data.shape)
print("Speakers:", data["speaker_id"].nunique())
print("Utterances:", data["utterance_id"].nunique())
print("Label counts:", data["intent_label"].value_counts())
```

**Step 2 – Build X, y, groups**

```python
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]

X = data[feature_columns]           # acoustic features
y = data["intent_label"]           # 0/1 trust label
groups = data["speaker_id"]        # speaker IDs for LOSO
```

**Step 3 – Set up LOSO**

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
```

**Step 4 – Loop over speakers; train two models per fold**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

log_accs, log_precs, log_recs, log_f1s = [], [], [], []
rf_accs,  rf_precs,  rf_recs,  rf_f1s  = [], [], [], []

fold_index = 0

for train_idx, test_idx in logo.split(X, y, groups):
    fold_index += 1
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_speakers = groups.iloc[test_idx].unique()
    
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    
    log_acc = accuracy_score(y_test, y_pred_log)
    log_prec = precision_score(y_test, y_pred_log)
    log_rec = recall_score(y_test, y_pred_log)
    log_f1 = f1_score(y_test, y_pred_log)
    log_accs.append(log_acc); log_precs.append(log_prec)
    log_recs.append(log_rec); log_f1s.append(log_f1)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_accs.append(rf_acc); rf_precs.append(rf_prec)
    rf_recs.append(rf_rec); rf_f1s.append(rf_f1)
    
    print(f"Fold {fold_index}, test speaker(s): {list(test_speakers)}")
    print("  Logistic Regression: acc", log_acc, "F1", log_f1)
    print("  Random Forest:       acc", rf_acc, "F1", rf_f1)
    print()
```

**Step 5 – Aggregate metrics across all speakers**

```python
log_accs, log_precs, log_recs, log_f1s = map(np.array, [log_accs, log_precs, log_recs, log_f1s])
rf_accs,  rf_precs,  rf_recs,  rf_f1s  = map(np.array, [rf_accs,  rf_precs,  rf_recs,  rf_f1s])

print("=== Logistic Regression (LOSO) ===")
print("Mean accuracy:", log_accs.mean(), "Std:", log_accs.std())
print("Mean F1:",      log_f1s.mean(),   "Std:", log_f1s.std())
print()

print("=== Random Forest (LOSO) ===")
print("Mean accuracy:", rf_accs.mean(), "Std:", rf_accs.std())
print("Mean F1:",      rf_f1s.mean(),   "Std:", rf_f1s.std())
print()
```

**Interpretation:**

* Higher mean F1 = better overall balance of precision and recall across speakers.
* Comparing LR vs RF:

  * If RF > LR, it suggests non-linear interactions between features are helpful.
  * If LR ≈ RF, the decision boundary may be mostly linear, or data is simple.

This is very similar to what you’d do to **reproduce the modeling section of the paper**, just with the real dataset and acoustics.

---

#### 6. Practice Exercises

**Exercise 1 – Why Multiple Models?**

Explain in 2–3 sentences why we test both Logistic Regression and Random Forest instead of only one model.

---

**Exercise 2 – Metrics Table**

Suppose after LOSO we get:

* Logistic Regression: mean F1 = 0.68, std F1 = 0.07
* Random Forest: mean F1 = 0.74, std F1 = 0.15

a) Which model has better average F1?
b) Which model’s performance varies more across speakers?

---

**Exercise 3 – Groups in LOSO**

What would happen if we used `groups = utterance_id` instead of `speaker_id` in `LeaveOneGroupOut`?

---

**Exercise 4 – Model Choice**

If your goal is to write a simple, interpretable baseline for a paper, which model would you choose first and why: Logistic Regression or Random Forest?

---

**Exercise 5 – Adding Features**

Name two additional kinds of features (besides pitch and HNR) that could be added to the pipeline in a more advanced version of this project.

---

#### 7. Solutions / Model Answers

**Exercise 1 – Sample Answer**

We test multiple models to see whether a simple linear boundary (Logistic Regression) is enough or whether a more flexible model (Random Forest) captures additional patterns. This helps us understand both **model capacity** and **dataset complexity**, instead of assuming one model is automatically best.

---

**Exercise 2 – Answer**

a) Random Forest has better average F1 (0.74 vs 0.68).
b) Random Forest has higher variation (std 0.15 vs 0.07), so its performance changes more across speakers.

---

**Exercise 3 – Answer**

Using `groups = utterance_id` would make each utterance its own group, so LOSO would hold out one **utterance** at a time rather than one speaker. That would **not** enforce speaker-independence: the same speaker would still appear in both training and test sets, and we would risk optimistic performance due to speaker-specific patterns.

---

**Exercise 4 – Answer**

For a simple, interpretable baseline, I’d choose **Logistic Regression** first. It’s easy to implement, fast to train, and its coefficients can be interpreted to see how each feature affects the log-odds of trust. Then I’d add Random Forest as a stronger non-linear model for comparison.

---

**Exercise 5 – Sample Answer**

Examples of additional features:

* **Prosody:** speaking rate, pause counts, energy (loudness) patterns.
* **Spectral features:** MFCCs (Mel-frequency cepstral coefficients), spectral centroid.
* **Linguistic features:** simple word counts or sentiment scores, if we have transcripts.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why do we use LOSO instead of a random 80/20 split here?
   **A:** Because we want to test generalization to **new speakers**, not just new utterances from the same speakers. LOSO enforces that.

2. **Q:** Could we include `mean_rating` itself as a feature?
   **A:** For pure trust prediction from acoustics, we usually don’t, because mean_rating is derived from human judgments. But we *can* use it to explore relationships between ratings and acoustics.

3. **Q:** Is Random Forest always better than Logistic Regression?
   **A:** No. Sometimes the data is simple and a linear model works just as well or better. RF is more flexible but can overfit or vary more across folds.

4. **Q:** Why do we track both mean and standard deviation of metrics across folds?
   **A:** Mean tells us overall performance; std tells us how stable performance is across speakers.

5. **Q:** Why do we use F1-score in addition to accuracy?
   **A:** Because accuracy can be misleading if classes are imbalanced. F1 balances precision and recall, especially important when wrong “trust” predictions are expensive.

6. **Q:** What if some speakers have very few utterances?
   **A:** Their folds may give unstable metrics. In real research, we might enforce a minimum number of utterances per speaker or analyze them separately.

7. **Q:** How close does this pipeline get to the actual paper?
   **A:** Conceptually it’s very close: same idea of speaker-independent evaluation, similar models, similar metrics. The main differences are that the real paper uses the **actual** dataset and may include more features and models.

8. **Q:** Could we add ROC-AUC to this pipeline?
   **A:** Yes. We’d have the models output probability scores and then compute ROC-AUC using those, per fold and averaged.

9. **Q:** Do we need to standardize features for Logistic Regression?
   **A:** In many cases, yes, especially if features are on very different scales. In this simple synthetic example, we sometimes skip it for teaching, but in the real reproduction we’d likely use `StandardScaler`.

10. **Q:** Is this pipeline only for trust, or could we use it for other emotions?
    **A:** The same structure works for many speech tasks: friendliness, dominance, politeness, emotion categories, etc.

---

#### 9. Quiz (10 Questions + Answers + Explanations)

**Q1.** The main purpose of using `speaker_id` as `groups` in LOSO is to:
a) Balance class labels
b) Ensure test speakers are unseen during training
c) Make the code faster
d) Increase the number of folds

**Answer:** b)
**Explanation:** Groups control how data is split; using speaker_id ensures speaker-independent evaluation.

---

**Q2.** In this lesson’s pipeline, each LOSO fold tests on:
a) One random utterance
b) All utterances from one speaker
c) Half the dataset
d) Only trustworthy utterances

**Answer:** b)
**Explanation:** Each group = one speaker; LOSO leaves out one group per fold.

---

**Q3.** Why do we compute metrics for each fold and then average them?
**Answer:** To summarize performance across all speakers and reduce the effect of any single “easy” or “hard” speaker.

---

**Q4.** True or False: “If Random Forest has a slightly higher mean F1 but much higher std, it might be less reliable across speakers than Logistic Regression.”

**Answer:** True.
**Explanation:** Higher std means more variability; some speakers may be predicted much worse.

---

**Q5.** Which model is more likely to capture complex non-linear interactions between features?
a) Logistic Regression
b) Random Forest

**Answer:** b)
**Explanation:** Random Forest builds many decision trees and can represent non-linear decision boundaries.

---

**Q6.** What does `max_iter=1000` do in Logistic Regression?
**Answer:** It increases the maximum number of optimization steps allowed so that the algorithm is more likely to converge.

---

**Q7.** Why is F1-score useful for trust vs neutral classification?
**Answer:** It balances precision and recall, which is important when misclassifying “trustworthy” vs “not trustworthy” has asymmetrical costs.

---

**Q8.** If the label counts are very imbalanced (e.g., many more 0s than 1s), what might we consider doing?
**Answer:** We might adjust class weights, collect more data, change the threshold, or use evaluation metrics that handle imbalance better (like F1 or ROC-AUC).

---

**Q9.** In this pipeline, what does `n_estimators=200` mean for Random Forest?
**Answer:** The forest consists of 200 decision trees, which usually improves stability and performance.

---

**Q10.** Why is this lesson crucial for reproducing the paper?
**Answer:** Because it shows how to implement the **complete modeling and evaluation process** that the paper uses, making it much easier to swap in the actual dataset and compare results.

---

#### 10. Mini Practice Project – End-to-End Trust Pipeline (Downloadable .zip)

**Mini Project Title:** *Speaker-Independent Trust Classification Pipeline*

I’ve prepared a mini project folder with:

* `synthetic_trust_full_pipeline_dataset.csv`

  * Columns:

    * `utterance_id`, `speaker_id`
    * `duration_seconds`, `mean_f0_hz`, `sd_f0_hz`, `hnr_db`, `shimmer_db`, `cpp_db`
    * `mean_rating`, `std_rating`, `n_ratings`
    * `intent_label` (0 = neutral/low trust, 1 = trustworthy)

* `full_trust_pipeline_loso.py`

  * Fully commented script that:

    * Loads data and prints basic info.
    * Builds X, y, groups.
    * Runs LOSO over speakers.
    * Trains Logistic Regression and Random Forest in each fold.
    * Computes accuracy, precision, recall, F1 per fold.
    * Aggregates metrics (mean and std) for each model.
    * Prints a short comparison summary.

* `README.txt` with instructions.

**Suggested student workflow:**

1. Unzip the file.

2. Install dependencies (if needed):

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Run:

   ```bash
   python full_trust_pipeline_loso.py
   ```

4. Answer:

   * Which model has higher mean F1?
   * Which has higher variability across speakers?
   * Does this match your expectations from earlier lessons?

5. Optional:

   * Add feature standardization for Logistic Regression.
   * Try changing `n_estimators` in Random Forest.
   * Add ROC-AUC computation.

---

#### 11. References

* scikit-learn docs:

  * `LeaveOneGroupOut`
  * `LogisticRegression`
  * `RandomForestClassifier`
  * `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
* Tutorials on:

  * “Building an end-to-end ML pipeline.”
  * “Group cross-validation and speaker-independent evaluation.”

---

#### 12. Additional Information / Teacher Tips

* **Mapping synthetic → real**
  Emphasize to the student that this synthetic project is a **dress rehearsal**. When they switch to the real paper’s dataset:

  * `synthetic_trust_full_pipeline_dataset.csv` → real features + labels file(s), joined.
  * Same LOSO logic, same metrics, same comparisons.

* **Encourage documentation**
  Get the student to write a short **mini-report**:

  * Data description
  * Methods (LOSO, models)
  * Results table (LR vs RF)
  * Interpretation

* **Prepare for Lesson 10**
  Next lesson can focus on:

  * Reproducing key numbers from the paper (as close as possible).
  * Writing up results in a **paper-style summary** (figures, tables, discussion).