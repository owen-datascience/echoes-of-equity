# ⭐ Lesson 5 — Evaluation Metrics, Confusion Matrices & Fairness Gaps

---

## 1. Lesson Summary

In this lesson, the student will:

* Learn the main **classification metrics**: accuracy, precision, recall, F1, ROC, AUC.
* Understand how to build and interpret a **confusion matrix**.
* Compute metrics **per demographic group** and measure **fairness gaps**.
* Connect metrics to the project goal: **reducing bias** across White, Black, and South Asian speakers.

By the end, the student will be able to clearly explain, with tables and plots, how well a model performs overall *and* how fair it is.

---

## 2. Key Points

* Accuracy alone can be misleading, especially with **imbalanced data**.
* **Confusion matrices** reveal the types of errors a model makes.
* **Precision** and **recall** describe different aspects of performance.
* **F1-score** balances precision and recall.
* **ROC curve** and **AUC** show performance at all thresholds, not just 0.5.
* Per-group metrics (by ethnicity) are essential for evaluating **demographic bias**.
* **Fairness gap** = difference in a metric (e.g., recall) between the best and worst group.
* Clear visualizations (tables, ROC plots, bar charts) are crucial for ISEF-level communication.

---

## 3. Real-World Examples or Stories

* A healthcare AI might have **95% accuracy overall**, but if recall for a minority group is only **70%**, that group is seriously underserved.
* The original trustworthiness paper reports different recall values for **White vs South Asian speakers**; your project’s goal is to **close that gap**, not just bump up overall accuracy. 
* Face recognition and recidivism prediction systems became controversial precisely because subgroup metrics revealed **large fairness gaps**.

Your student’s project will stand out at ISEF if they show *both* overall performance and careful fairness analysis.

---

## 4. Terminology Explained

* **True Positive (TP)** – Model predicts “trustworthy” and it is truly trustworthy.
* **True Negative (TN)** – Model predicts “neutral” and it is truly neutral.
* **False Positive (FP)** – Model predicts “trustworthy” but it’s actually neutral.
* **False Negative (FN)** – Model predicts “neutral” but it’s actually trustworthy.
* **Confusion Matrix** – A 2×2 table of TP, FP, FN, TN for binary classification.
* **Accuracy** – (TP + TN) / (TP + TN + FP + FN).
* **Precision (Trustworthy)** – TP / (TP + FP) → “Of what I predicted as trustworthy, how many were correct?”
* **Recall (Trustworthy)** – TP / (TP + FN) → “Of all truly trustworthy, how many did I catch?”
* **F1 Score** – Harmonic mean of precision and recall.
* **ROC Curve** – Plot of True Positive Rate vs False Positive Rate across thresholds.
* **AUC (Area Under the ROC Curve)** – Single number summarizing ROC performance (1.0 = perfect, 0.5 = random).
* **Fairness Gap** – Difference between best and worst subgroup metric (e.g., recall).

---

## 5. How It Works (Step-by-Step)

Assume you have:

* `y_true` – true labels (0 = neutral, 1 = trustworthy).
* `y_pred` – predictions at threshold 0.5.
* `score_trustworthy` – predicted probability/logit for “trustworthy”.
* `ethnicity` – `White`, `Black`, `SouthAsian`.

### A. Overall Confusion Matrix & Metrics

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

cm = confusion_matrix(y_true, y_pred)
print(cm)  # [[TN, FP],
           #  [FN, TP]]

print(classification_report(y_true, y_pred, target_names=["neutral","trustworthy"]))
```

### B. ROC Curve & AUC

```python
fpr, tpr, thresholds = roc_curve(y_true, score_trustworthy)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)
```

Plot with matplotlib to visualize trade-offs.

### C. Metrics by Ethnicity

```python
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

df = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred,
    "score_trustworthy": score_trustworthy,
    "ethnicity": ethnicity
})

group_recalls = {}
for group, sub in df.groupby("ethnicity"):
    rec = recall_score(sub["y_true"], sub["y_pred"], pos_label=1)
    prec = precision_score(sub["y_true"], sub["y_pred"], pos_label=1)
    f1 = f1_score(sub["y_true"], sub["y_pred"], pos_label=1)
    group_recalls[group] = rec
    print(group, "Precision:", prec, "Recall:", rec, "F1:", f1)
```

### D. Fairness Gap

```python
max_rec = max(group_recalls.values())
min_rec = min(group_recalls.values())
gap = max_rec - min_rec
print("Fairness gap in recall:", gap)
```

The student can later compare **Random Forest vs CNN vs Adversarial CNN** using these metrics, showing how fairness improves.

---

## 6. Practice Exercises (5)

**Exercise 1 – Manually Compute Metrics from a Confusion Matrix**
Given a confusion matrix:

[
\begin{bmatrix}
TN & FP \
FN & TP
\end{bmatrix}
=============

\begin{bmatrix}
40 & 10 \
5 & 45
\end{bmatrix}
]

Compute accuracy, precision (trustworthy), recall (trustworthy), and F1.

---

**Exercise 2 – Confusion Matrix in Python**
Write code using scikit-learn to print a confusion matrix and classification report for given `y_true` and `y_pred`.

---

**Exercise 3 – ROC & AUC**
Given arrays `y_true` and `score_trustworthy`, use scikit-learn to:

1. Compute `fpr`, `tpr`, `thresholds`.
2. Compute `auc(fpr, tpr)`.
3. Plot the ROC curve.

---

**Exercise 4 – Per-Group Recall**
Using a DataFrame with `y_true`, `y_pred`, and `ethnicity`, calculate recall for each group and print them.

---

**Exercise 5 – Fairness Gap & Interpretation**
Using the per-group recalls from Exercise 4:

1. Compute the fairness gap.
2. Write 3–4 sentences interpreting what this gap means and how it might affect real users.

---

## 7. Solutions (Model Answers)

**Solution 1 – Manual Metrics**

From:

* TN = 40, FP = 10, FN = 5, TP = 45

Total = 40 + 10 + 5 + 45 = 100

* Accuracy = (TP + TN) / Total = (45 + 40) / 100 = **0.85**
* Precision (trustworthy) = TP / (TP + FP) = 45 / (45 + 10) = 45/55 ≈ **0.818**
* Recall (trustworthy) = TP / (TP + FN) = 45 / (45 + 5) = 45/50 = **0.90**
* F1 = 2 * (P * R) / (P + R) ≈ 2 * (0.818 * 0.90) / (0.818 + 0.90) ≈ **0.857**

---

**Solution 2 – Confusion Matrix in Python**

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["neutral", "trustworthy"]))
```

---

**Solution 3 – ROC & AUC**

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true, score_trustworthy)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

---

**Solution 4 – Per-Group Recall**

```python
import pandas as pd
from sklearn.metrics import recall_score

df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "ethnicity": ethnicity})

for group, sub in df.groupby("ethnicity"):
    rec = recall_score(sub["y_true"], sub["y_pred"], pos_label=1)
    print(group, "Recall(trustworthy):", rec)
```

---

**Solution 5 – Fairness Gap & Interpretation**

```python
recalls = {}
for group, sub in df.groupby("ethnicity"):
    rec = recall_score(sub["y_true"], sub["y_pred"], pos_label=1)
    recalls[group] = rec

max_rec = max(recalls.values())
min_rec = min(recalls.values())
gap = max_rec - min_rec
print("Fairness gap:", gap)
```

Interpretation example:

> If recall is 0.85 for White speakers and 0.70 for South Asian speakers, the gap is 0.15 (15 percentage points). That means South Asian speakers are 15% more likely to have their trustworthy speech missed by the model. In real-world use (like public speaking coaching or hiring), this could systematically disadvantage them.

---

## 8. Q&A (10 Common Questions)

1. **Q:** Why isn’t accuracy enough?
   **A:** Because it hides how errors are distributed. A model could be very accurate overall but still perform poorly on a minority group.

2. **Q:** When is recall more important than precision?
   **A:** When **missing positives** is worse than having some false alarms—for example, failing to detect trustworthy speech for certain demographics.

3. **Q:** What does a point on the ROC curve represent?
   **A:** The (FPR, TPR) pair at a specific decision threshold.

4. **Q:** Can AUC be high even if fairness is bad?
   **A:** Yes. A model might rank samples well overall but still treat some subgroups worse.

5. **Q:** What threshold should I use?
   **A:** 0.5 is common, but you can tune it based on the trade-off between precision and recall. You can also compare subgroup performance at your chosen threshold.

6. **Q:** Why focus on recall for “trustworthy” rather than “neutral”?
   **A:** The project’s risk is missing people who are actually trustworthy (false negatives), which can harm minority speakers more.

7. **Q:** What is macro vs weighted F1?
   **A:** Macro F1 averages F1 over classes equally; weighted F1 weights by class frequency.

8. **Q:** How do judges like to see metrics presented?
   **A:** Confusion matrices, metric tables, and ROC curves, plus clear explanations of what improved and how fairness gaps changed.

9. **Q:** Can fairness gap ever be 0?
   **A:** In theory yes, but in practice small gaps are fine; your goal is to **reduce** large gaps from the baseline.

10. **Q:** How does this connect to adversarial debiasing later?
    **A:** These metrics are how you **prove** that adversarially trained models reduce subgroup gaps compared to non-debiased models.

---

## 9. Quiz (10 Questions)

1. **Which of the following is TP?**
   A) Model predicts neutral, true label neutral
   B) Model predicts trustworthy, true label trustworthy ✅
   C) Model predicts neutral, true label trustworthy
   D) Model predicts trustworthy, true label neutral

2. **Accuracy formula:**
   ➜ (TP + TN) / (TP + TN + FP + FN)

3. **Recall (trustworthy) measures:**
   ➜ Of all truly trustworthy samples, how many the model correctly labels as trustworthy.

4. **Precision (trustworthy) measures:**
   ➜ Of all samples predicted trustworthy, how many are truly trustworthy.

5. **What metric balances precision and recall?**
   ➜ F1-score.

6. **On an ROC curve, the x-axis is:**
   ➜ False Positive Rate.

7. **AUC of 0.5 means:**
   ➜ The model is performing like random guessing.

8. **A fairness gap in recall of 0.20 means:**
   ➜ The best group’s recall is 20 percentage points higher than the worst group’s.

9. **Which function computes an ROC curve in scikit-learn?**
   ➜ `roc_curve`.

10. **Why compute metrics per ethnicity?**
    ➜ To detect and measure demographic bias in the model.

---

## 10. Mini Practice Project – Evaluation & Fairness Metrics (with .zip)

I prepared a mini project so the student can *actually* compute these metrics on a small synthetic dataset.

### Project Goal

* Load a small CSV with **true labels**, **predicted scores**, **hard predictions**, and **ethnicity**.
* Compute confusion matrix, precision, recall, F1, ROC curve, AUC.
* Compute per-ethnicity metrics and fairness gap in recall.
* Save the ROC curve as an image.

### What’s in the .zip

**`lesson5_miniproject_eval.zip`** contains:

* `lesson5_miniproject_eval/`

  * `README.md` – Instructions.
  * `data/synthetic_eval_results.csv` – Columns:

    * `y_true` (0/1)
    * `score_trustworthy` (0–1)
    * `y_pred` (0/1 at threshold 0.5)
    * `ethnicity` (`White`, `Black`, `SouthAsian`)
  * `src/analyze_metrics.py` – Script that:

    * Loads the CSV
    * Prints confusion matrix & classification report
    * Computes ROC & AUC and saves `data/roc_overall.png`
    * Computes per-ethnicity precision/recall/F1
    * Prints fairness gap in recall

### How to Run

1. Unzip the archive.

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. Run:

   ```bash
   python src/analyze_metrics.py
   ```

4. Check the printed metrics and open `data/roc_overall.png`.

5. Have the student write a **short paragraph** summarizing:

   * Overall accuracy & AUC
   * Recall per group
   * Fairness gap and what it means

This paragraph can later be adapted directly into the ISEF paper’s **“Baseline Evaluation”** section.

---

## 11. References

* Project timeline sections: “Replication – Confusion Matrix” and “Final Analysis & Presentation.” 
* scikit-learn metrics documentation: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
* ROC & AUC explanation: any standard ML blog/tutorial (e.g., scikit-learn’s ROC tutorial).
* Papers and tutorials on fairness metrics (demographic parity, equalized odds) for further reading.

---

## 12. Additional Information (Mentor Tips)

* Encourage the student to **store all confusion matrices and ROC curves** for each model version (Random Forest, CNN, adversarial CNN). This history becomes powerful evidence of improvement.
* For the final poster, I’d suggest:

  * One **overall ROC curve** comparing different models.
  * A **bar chart** of recall per ethnicity for baseline vs debiased model.
* Emphasize that *clear evaluation* is what differentiates a “cool project” from a **publishable, competition-winning study**.