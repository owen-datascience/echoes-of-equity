### Lesson 10 – Mini Replication Study & Writing Up Results

This is the “capstone” lesson: your student now acts like a **junior researcher** reproducing a simplified version of the paper’s pipeline and **writing a mini report** about it.

---

#### 1. Lesson Summary

In this final lesson, the student:

* Runs a **full trust-detection experiment** on a synthetic dataset that mimics the paper’s structure.
* Uses **speaker-independent LOSO evaluation** with Logistic Regression and Random Forest.
* Computes **accuracy, precision, recall, F1, and confusion matrices**.
* Generates and reads a **short automatic report**.
* Writes their own **mini replication report** describing goals, methods, results, and interpretation.

This gives them a rehearsal for reproducing and reporting the results of the real paper: they’ll know not just how to code, but how to **tell the research story** clearly.

---

#### 2. Key Points

* A replication is not just about running code; it’s about:

  * Designing a clear **experiment**.
  * Evaluating fairly.
  * **Summarizing** results in words, tables, and maybe plots.
* The structure of a typical report:

  * **Goal** → **Data** → **Methods** → **Results** → **Interpretation** → **Limitations & Future work**.
* Confusion matrices help understand **error types**:

  * False positives (predicting trust but clip is not).
  * False negatives (missing trust).
* The pipeline is now “modular”:

  * Dataset can be swapped with the real paper’s dataset.
  * Models can be changed/extended.
  * Evaluation strategy (LOSO) stays the same.
* Good scientific practice means:

  * Being **honest** about what works and what doesn’t.
  * Reporting **both mean and variability**.
  * Relating results back to **theory and prior work** (here, the paper).

---

#### 3. Real-World Examples or Stories

1. **Science fair project report**
   A strong science fair project doesn’t just show a cool device; it includes a **logbook**, organized **results**, and clear **conclusions**. That’s exactly what you’re training the student to do here.

2. **Company internal experiments**
   A product team tests a new voice feature and writes an **internal experiment doc** with data, methodology, metrics, and “Ship or not?” This lesson is essentially that process for trust detection.

3. **Academic replication**
   In real research, teams try to **replicate** published work on different datasets or with slightly different methods. Your student is learning the mindset and skills to do that responsibly.

---

#### 4. Terminology Explained

* **Replication study** – An experiment that attempts to reproduce results or patterns from a previous paper, sometimes with new data or simplified settings.
* **Confusion matrix** – A 2×2 table showing counts of:

  * True Negative (TN), False Positive (FP)
  * False Negative (FN), True Positive (TP)
* **False positive (FP)** – Model predicts “trustworthy” when it’s actually neutral/low trust.
* **False negative (FN)** – Model predicts “neutral/low trust” when clip is actually trustworthy.
* **Experiment report** – A structured document explaining what you did, why, how, and what you found.
* **Template** – A skeleton structure to help the student organize their report.

---

#### 5. How It Works – Mini Replication Experiment

We now use a synthetic dataset `replication_trust_dataset.csv` with columns:

```text
utterance_id | speaker_id | duration_seconds | mean_f0_hz | sd_f0_hz
hnr_db | shimmer_db | cpp_db | mean_rating | std_rating | n_ratings | intent_label
```

**Step 1 – Load and inspect**

```python
import pandas as pd

data = pd.read_csv("replication_trust_dataset.csv")
print(data.head(6))
print("Shape:", data.shape)
print("Speakers:", data["speaker_id"].nunique())
print("Utterances:", data["utterance_id"].nunique())
print("Label counts:", data["intent_label"].value_counts())
```

**Step 2 – Build X, y, groups**

```python
feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]

X = data[feature_columns]
y = data["intent_label"]
groups = data["speaker_id"]
```

**Step 3 – LOSO + two models + metrics + confusion matrices**

The script:

* Uses `LeaveOneGroupOut` with `groups = speaker_id`.
* In each fold:

  * Trains Logistic Regression and Random Forest.
  * Computes accuracy, precision, recall, F1.
  * Stores predictions for a **global confusion matrix**.
* After all folds:

  * Computes mean and std of metrics.
  * Builds confusion matrices for both models.
  * Writes everything to `replication_report.txt` (a mini machine-written report).

You (or the student) then open that file and fill in a more polished version using `replication_report_template.txt`.

---

#### 6. Practice Exercises

**Exercise 1 – Reading the Auto Report**

After running the script, open `replication_report.txt`.
a) Write down the mean F1 for Logistic Regression and Random Forest.
b) Which model performed better overall?

---

**Exercise 2 – Confusion Matrix Interpretation**

Suppose the Random Forest confusion matrix in the report is:

```text
[[80 20]
 [18 94]]
```

Rows = true labels, columns = predicted labels.
a) How many true neutral/low trust clips were misclassified as trustworthy?
b) How many true trustworthy clips were misclassified as neutral/low trust?

---

**Exercise 3 – Error Type Discussion**

Using the confusion matrix above:
If your system will be used to **screen audio for trustworthy-sounding voices**, which type of mistake is worse, FP or FN, and why? Answer in 2–3 sentences.

---

**Exercise 4 – Speaker Variability**

In the report, look at the std of F1 scores across folds for Random Forest.
a) Is it small (e.g., < 0.05) or large (e.g., > 0.15)?
b) What does that tell you about how performance differs across speakers?

---

**Exercise 5 – Mapping to the Real Paper**

List three steps from this synthetic replication that would remain **the same** when the student uses the real dataset and code from *“Human voices communicating trustworthy intent”*, and two steps that would change.

---

#### 7. Solutions / Model Answers

(These are model answers; your student’s numbers will depend on their run.)

**Exercise 1 – Sample Answer**

a) Example (your actual numbers may differ):

* Logistic Regression mean F1 ≈ 0.70
* Random Forest mean F1 ≈ 0.77

b) Random Forest performed better overall because its mean F1 is higher.

---

**Exercise 2 – Answer**

Matrix:

```text
[[80 20]
 [18 94]]
```

* Row 0, Col 1 = 20 → these are **false positives** (true 0, predicted 1).
* Row 1, Col 0 = 18 → these are **false negatives** (true 1, predicted 0).

a) 20 neutral/low trust clips misclassified as trustworthy.
b) 18 trustworthy clips misclassified as neutral/low trust.

---

**Exercise 3 – Sample Answer**

If the system is used to screen for trustworthy voices, **false positives** might be more dangerous because you are incorrectly treating untrustworthy or neutral voices as trustworthy. That could lead to over-trusting someone. However, in some applications (e.g., trying not to offend people by labeling them “untrustworthy”), false negatives might also be serious. The “worse” error depends on the real-world context, which should be discussed in the project.

---

**Exercise 4 – Sample Answer**

a) If std(F1) is large (e.g., > 0.15), then performance changes a lot from speaker to speaker.
b) That suggests the model works well for some speakers but struggles for others. In a real replication, we might inspect which speakers are hard and why (accent, speaking style, recording quality, etc.).

---

**Exercise 5 – Sample Answer**

Same steps (synthetic vs real paper):

1. Using utterance-level labels derived from human trust ratings.
2. Using acoustic features (e.g., pitch, HNR, CPP) as model inputs.
3. Applying speaker-independent LOSO evaluation with similar metrics (accuracy, precision, recall, F1).

Different steps:

1. The real paper uses **actual raw audio + feature extraction** pipelines instead of pre-made synthetic CSVs.
2. The real replication will use the authors’ **exact feature definitions, thresholds, and modeling details**, possibly including more advanced models or regularization settings.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Is this synthetic replication “good enough” to say we replicated the paper?
   **A:** It’s a **training replica**: it mirrors the structure and logic of the paper on fake data. The real replication still requires running the pipeline on the actual dataset.

2. **Q:** Why do we write a report instead of just showing metrics in the terminal?
   **A:** Writing a report trains you to communicate results clearly, which is essential for competitions, papers, and internships.

3. **Q:** Should we ever report just one metric (like accuracy)?
   **A:** No. It’s better to include multiple metrics (accuracy, precision, recall, F1, confusion matrix) to capture different aspects of performance.

4. **Q:** What if Random Forest is only slightly better than Logistic Regression?
   **A:** That’s okay! It still tells you that a simple model is competitive, which is scientifically interesting.

5. **Q:** Could this same pipeline be extended to multi-class labels?
   **A:** Yes. You’d change the labels and use multi-class metrics, but the overall structure (features, LOSO, models) remains similar.

6. **Q:** Do I have to use LOSO forever?
   **A:** No, but for **speaker traits** like trust, LOSO is a strong, fair evaluation choice. Other tasks may need different strategies.

7. **Q:** How do I handle randomness in results?
   **A:** Fix random seeds where possible, and report averages. Slight variation is normal; big variation is a signal to investigate.

8. **Q:** How can I make my replication more similar to the actual paper?
   **A:** Use the real dataset, match the exact feature set, match model types and parameters, and compare your reported numbers with theirs.

9. **Q:** Is it okay if my numbers don’t exactly match the paper’s?
   **A:** Yes. Replications often differ. The key is to **document what you did** and explore possible reasons for differences.

10. **Q:** How does this help with ISEF or other competitions?
    **A:** It trains you to design proper experiments, evaluate them correctly, and present them like a real scientist—skills judges care about a lot.

---

#### 9. Quiz (10 Questions + Answers + Explanations)

**Q1.** A replication study mainly aims to:
a) Invent a brand-new idea
b) Reproduce and test existing findings
c) Avoid writing reports
d) Remove human ratings

**Answer:** b)
**Explanation:** Replication checks whether previous findings hold under similar or slightly different conditions.

---

**Q2.** A confusion matrix with many false positives means the model often:
**Answer:** Predicts “trustworthy” when the true label is neutral/low trust.

---

**Q3.** True or False: “In speaker-independent evaluation, it’s okay to have the same speaker in both train and test sets.”

**Answer:** False.
**Explanation:** That defeats the purpose; we want test speakers to be unseen.

---

**Q4.** Which is NOT typically included in an experiment report?
a) Goal
b) Methods
c) Random song lyrics
d) Results

**Answer:** c)

---

**Q5.** Why do we save an automatic text report (`replication_report.txt`)?
**Answer:** To capture key results in a permanent, readable format that the student can analyze and use when writing their own report.

---

**Q6.** If Random Forest has higher F1 but also much higher std across folds than Logistic Regression, what might you say?
**Answer:** RF is stronger on average but less stable; some speakers may be much better or worse than with LR.

---

**Q7.** In this mini replication, what do we treat as the “ground truth”?
**Answer:** The binary trust labels (`intent_label`) derived from mean human trust ratings.

---

**Q8.** What is one reason to include confusion matrices in your report?
**Answer:** They show how the model’s errors are distributed across classes (FP vs FN), not just overall accuracy.

---

**Q9.** True or False: “Once you have a working pipeline, replacing the synthetic dataset with the real dataset is usually straightforward.”

**Answer:** True.
**Explanation:** Most of the code structure stays the same; you mainly change file paths and feature extraction.

---

**Q10.** Why is this lesson a good final step in the course?
**Answer:** Because it requires combining everything learned—data handling, labeling, modeling, evaluation, and reporting—into a realistic mini research project.

---

#### 10. Mini Practice Project – Mini Replication Study

**Mini Project Title:** *Mini Replication: Voice Trust Classification*

I’ve prepared a complete .zip package that includes:

* `replication_trust_dataset.csv`
* `replication_trust_experiment.py` – fully commented pipeline
* `replication_report_template.txt` – write-up template
* `README.txt` – instructions

**Student instructions:**

1. Unzip the folder.
2. Install requirements (if needed):

   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the experiment:

   ```bash
   python replication_trust_experiment.py
   ```
4. Open `replication_report.txt` and read the summary.
5. Use `replication_report_template.txt` to write your own 1–3 page mini report:

   * Describe data, methods, results, and interpretation.
   * Reflect on how this connects to the real paper.

---

#### 11. References

* scikit-learn documentation on:

  * `LeaveOneGroupOut`
  * `LogisticRegression`
  * `RandomForestClassifier`
  * `confusion_matrix` and other metrics
* General guides on:

  * “How to write a lab report”
  * “How to write a machine learning experiment report”

---

#### 12. Additional Information / Teacher Tips

* You can now **swap in the real dataset** from *“Human voices communicating trustworthy intent”* and adapt this pipeline:

  * Replace synthetic CSV with real utterance-level feature + label files.
  * Match the paper’s exact features and thresholding choices.
  * Compare your student’s results to reported ones (even approximate comparison is instructive).

* For ISEF or similar competitions, this lesson sets the pattern for:

  * **Method section** (data, models, evaluation).
  * **Results section** (tables, metrics, plots).
  * **Discussion** (interpretation + limitations + future work).