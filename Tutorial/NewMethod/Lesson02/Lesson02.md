# ⭐ Lesson 2 — Exploring the Dataset & Building a Fairness-Aware Baseline

---

## 1. Lesson Summary

In this lesson, the student will:

* Find out **what’s actually inside the dataset** (utterances, speakers, demographics, labels).
* Learn how to **load and inspect** the data using pandas.
* Build a **simple “baseline” model** (majority-class or simple classifier) to measure starting performance.
* Compute **metrics by demographic group** (White, Black, South Asian) to see existing fairness gaps.

This is the bridge between the project plan and real code: by the end of Lesson 2, the student should be able to answer, with numbers, “How unfair is the starting point?”—a key question that ISEF judges will love. 

---

## 2. Key Points

* Always **understand your dataset** before building complex models.
* A **baseline** is a simple method you can beat (e.g., always predicting the majority class).
* Metrics like **accuracy** alone can hide important fairness problems.
* It’s crucial to look at **per-group performance** (e.g., recall for South Asian speakers).
* **pandas** is the main tool for exploring structured data.
* We can simulate fairness analysis even on a **small toy dataset** before using the full OSF dataset.
* Documenting early findings sets up your later “innovation” story.

---

## 3. Real-World Examples or Stories

* A hiring model that has **90% accuracy overall** might have **60% accuracy** for one demographic group. That’s unacceptable in real life but invisible if you only look at the average.
* Facial recognition systems with high overall accuracy have historically misclassified **darker-skinned women** much more often than lighter-skinned men. Only **group-wise metrics** revealed this.
* For your project, the original Random Forest model had better recall for **White speakers** than for **South Asian speakers**, exposing WWIB (White Western Individualist Bias). 

These stories motivate why Lesson 2 is not just “data cleaning”—it’s about **justice** and **rigor**.

---

## 4. Terminology Explained

* **Dataset** – A collection of data points; here, audio utterances + labels + demographics.
* **Label** – The target we want to predict; here: `trustworthy` vs `neutral`.
* **Feature** – The input used by a model (later: Mel-spectrogram pixels).
* **Exploratory Data Analysis (EDA)** – Systematic first look at data: shapes, counts, missing values, distributions.
* **Baseline Model** – A simple model used as a reference point (e.g., majority-class predictor).
* **Subgroup** – A subset of data defined by a shared attribute, like ethnicity.
* **Accuracy** – Fraction of predictions that are correct.
* **Recall** – Of all truly positive cases, how many the model correctly finds.
* **Fairness Gap** – Difference in a metric (like recall) between demographic groups.

---

## 5. How It Works (Step-by-Step)

### A. Load and Inspect the Data

1. Put the dataset (e.g., CSV with metadata) in a `data/` folder.

2. Use pandas to load:

   ```python
   import pandas as pd
   df = pd.read_csv("data/metadata.csv")
   print(df.head())
   print(df.columns)
   ```

3. Check basic info:

   ```python
   print(df.shape)
   print(df["ethnicity"].value_counts())
   print(df["intent_label"].value_counts())
   ```

### B. Build a Simple Baseline

* **Majority-class baseline:**
  Find the most common label, predict it for everyone.

  ```python
  majority_label = df["intent_label"].mode()[0]
  df["baseline_pred"] = majority_label
  ```

* Or a **very simple classifier** (e.g., logistic regression on numeric features if available).

### C. Compute Metrics Overall and by Group

```python
from sklearn.metrics import accuracy_score, recall_score

acc = accuracy_score(df["intent_label"], df["baseline_pred"])
rec = recall_score(df["intent_label"], df["baseline_pred"], pos_label="trustworthy")
print("Overall accuracy:", acc)
print("Overall recall (trustworthy):", rec)

for group, sub in df.groupby("ethnicity"):
    acc_g = accuracy_score(sub["intent_label"], sub["baseline_pred"])
    rec_g = recall_score(sub["intent_label"], sub["baseline_pred"], pos_label="trustworthy")
    print(group, "Accuracy:", acc_g, "Recall:", rec_g)
```

### D. Interpret Fairness Gaps

* If recall for **White** is 0.75 but for **SouthAsian** is 0.55 → **20% fairness gap**.
* This becomes your **“problem statement with numbers”** for your poster and paper.

---

## 6. Practice Exercises (5)

**Exercise 1 – Column Inspection**
Given a CSV with columns: `utterance_id, speaker_id, ethnicity, intent_label`, write code to:

1. Load the file.
2. Print the first 5 rows.
3. Print the unique values in `ethnicity` and `intent_label`.

---

**Exercise 2 – Distribution Counts**
Write code to:

1. Count how many rows there are for each ethnicity.
2. Count how many `trustworthy` vs `neutral` labels there are overall.

---

**Exercise 3 – Majority Baseline**

1. Find the majority class for `intent_label`.
2. Create a new column `baseline_pred` that always predicts this label.

---

**Exercise 4 – Overall Metrics**
Use scikit-learn to compute:

* Overall accuracy
* Overall recall for the `trustworthy` class

---

**Exercise 5 – Group Metrics & Fairness Gap**
For each ethnicity:

1. Compute accuracy and recall for `baseline_pred`.
2. Compute the **fairness gap** between the group with the highest recall and the lowest recall.

---

## 7. Solutions (Model Answers)

**Solution 1 – Column Inspection**

```python
import pandas as pd

df = pd.read_csv("data/mini_trust_dataset.csv")
print(df.head())
print("Ethnicities:", df["ethnicity"].unique())
print("Intent labels:", df["intent_label"].unique())
```

---

**Solution 2 – Distribution Counts**

```python
print("Counts per ethnicity:")
print(df["ethnicity"].value_counts())
print("\nLabel counts:")
print(df["intent_label"].value_counts())
```

---

**Solution 3 – Majority Baseline**

```python
majority_label = df["intent_label"].mode()[0]
print("Majority label:", majority_label)
df["baseline_pred"] = majority_label
```

---

**Solution 4 – Overall Metrics**

```python
from sklearn.metrics import accuracy_score, recall_score

acc = accuracy_score(df["intent_label"], df["baseline_pred"])
rec = recall_score(df["intent_label"], df["baseline_pred"], pos_label="trustworthy")

print(f"Overall accuracy: {acc:.3f}")
print(f"Overall recall (trustworthy): {rec:.3f}")
```

---

**Solution 5 – Group Metrics & Fairness Gap**

```python
group_recalls = {}

for group, sub in df.groupby("ethnicity"):
    acc_g = accuracy_score(sub["intent_label"], sub["baseline_pred"])
    rec_g = recall_score(sub["intent_label"], sub["baseline_pred"], pos_label="trustworthy")
    group_recalls[group] = rec_g
    print(f"{group}: Accuracy={acc_g:.3f}, Recall(trustworthy)={rec_g:.3f}")

max_rec = max(group_recalls.values())
min_rec = min(group_recalls.values())
gap = max_rec - min_rec
print(f"\nFairness gap in recall: {gap:.3f}")
```

A gap > 0.10 (10 percentage points) is already a red flag the student can mention.

---

## 8. Q&A (10 Questions)

1. **Q:** Why do we bother with a simple baseline?
   **A:** To have a clear “starting line” to beat. Judges want to see improvement over something, not just one fancy model.

2. **Q:** Is a high overall accuracy always good?
   **A:** Not necessarily. If one group is large, it can dominate the metric and hide poor performance for others.

3. **Q:** Why use pandas instead of plain Python lists?
   **A:** pandas gives powerful tools (groupby, value_counts, etc.) that make analysis easier and less error-prone.

4. **Q:** Why group by ethnicity?
   **A:** Because bias often appears as performance differences between demographic groups.

5. **Q:** What if the dataset is imbalanced (e.g., more trustworthy than neutral)?
   **A:** Then a majority-class baseline might look strong on accuracy but can have poor recall for the minority class.

6. **Q:** Should we always predict the majority class in real systems?
   **A:** No—it’s only for benchmarking. Real systems must treat minority cases fairly.

7. **Q:** Is recall the only fairness metric?
   **A:** No. Precision, F1, false positive rate, and other metrics can also matter, but recall is especially important when missing positives is harmful.

8. **Q:** Could we use accuracy for fairness gap instead of recall?
   **A:** Yes, but recall is more sensitive when the positive class is rare or important (like detecting trustworthy intent).

9. **Q:** Do I need the full OSF dataset to practice this lesson?
   **A:** No, we start with a small synthetic dataset (in the mini project) to learn the workflow.

10. **Q:** How do these analyses appear in an ISEF poster?
    **A:** Typically as bar charts comparing recall/accuracy across groups, plus a short text explaining the fairness gap.

---

## 9. Quiz (10 Questions)

**1. What is the purpose of a baseline model?**
**Answer:** To provide a simple reference level of performance that more advanced models should beat.

---

**2. Which Python library is most useful for EDA on tabular data?**
**Answer:** `pandas`.

---

**3. What is a majority-class baseline?**
**Answer:** A model that always predicts the most frequent label in the dataset.

---

**4. Why is looking only at overall accuracy dangerous?**
**Answer:** It can hide poor performance on smaller demographic groups.

---

**5. What function in pandas gives label counts?**
**Answer:** `value_counts()`.

---

**6. What is a subgroup in fairness analysis?**
**Answer:** A subset of data defined by attributes like ethnicity, gender, or age.

---

**7. Which metric answers: “Of all true trustworthy cases, how many did we catch?”**
**Answer:** Recall for the `trustworthy` class.

---

**8. How do you compute fairness gap in recall?**
**Answer:** Max recall across groups minus min recall across groups.

---

**9. What is one sign that your dataset may be biased?**
**Answer:** Large differences in performance metrics between demographic groups.

---

**10. Why is documenting early metrics important?**
**Answer:** It lets you prove later that your new model genuinely reduces bias and improves over the baseline.

---

## 10. Mini Practice Project — Baseline Fairness Analysis (with .zip)

### Goal

Use a **small synthetic dataset** to:

1. Load and inspect voice-intent data.
2. Build a majority-class baseline model.
3. Compute and interpret fairness metrics by ethnicity.

### What’s Included in the .zip

I’ve created a ready-to-use archive:

**`lesson2_miniproject_echoes_fairness.zip`** containing:

* `lesson2_miniproject_echoes/`

  * `README.md` – Instructions.
  * `data/mini_trust_dataset.csv` – Synthetic dataset with:

    * `utterance_id`
    * `speaker_id`
    * `ethnicity` (`White`, `Black`, `SouthAsian`)
    * `intent_label` (`trustworthy`, `neutral`)
  * `src/analyze_fairness.py` – Python script that:

    * Loads the dataset
    * Shows distributions
    * Builds a majority baseline
    * Computes accuracy & recall overall and by ethnicity

### How to Run

1. Unzip the file.

2. (Optional but recommended) create a virtual environment.

3. Install dependencies:

   ```bash
   pip install pandas scikit-learn
   ```

4. Run:

   ```bash
   python src/analyze_fairness.py
   ```

5. Look at the printed metrics and answer:

   * Which ethnicity has the highest recall for `trustworthy`?
   * Which has the lowest?
   * What is the fairness gap?
   * How might this motivate your later CNN + adversarial debiasing work?

---

## 11. References

* Project plan PDF (Echoes of Equity) for definitions, milestones, and fairness goals. 
* pandas documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
* scikit-learn metrics guide: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
* Fairness in ML overview: [https://developers.google.com/machine-learning/fairness-overview](https://developers.google.com/machine-learning/fairness-overview)

---

## 12. Additional Information (Mentor Tips)

* Encourage the student to **copy plots and tables from this mini project** into their research notebook—this becomes early evidence of bias.
* Have them write a **short paragraph** summarizing their fairness findings; later this becomes part of the ISEF paper’s “Baseline Results” section.
* Remind them: *“If you can clearly show the unfairness at the start, your later improvements will look much more impressive.”*