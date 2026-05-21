### Lesson 7 – Feature Importance & Interpreting the Trust Model

---

#### 1. Lesson Summary

In this lesson, the student learns **how to “look inside” a model** to see **which acoustic features matter most** for predicting trustworthy vs neutral intent.

We’ll focus on:

* Why interpretation is important in research (not just “does it work?” but **“why?”**).
* How **Random Forest feature importances** work.
* How to read and reason about:

  * Which features have high importance (e.g., pitch, HNR, CPP).
  * Which features matter less (e.g., maybe duration or shimmer).
* How to connect model importance back to ideas like:

  > “When people try to sound trustworthy, they tend to speak with clearer, more stable voices and slightly higher pitch.”

By the end, the student will be able to **train a Random Forest on acoustic features and interpret feature importances** to support or question hypotheses in the paper.

---

#### 2. Key Points

* A model that performs well is **good**, but a model we can **understand** is **better**—especially in research.
* **Feature importance** tells us which input features have the most influence on the model’s decisions.
* In **Random Forest**, importance is usually based on:

  * How much each feature reduces impurity across all trees.
* Importance is **relative**:

  * A feature with importance 0.4 is more influential than one with 0.1, but the absolute numbers are less meaningful.
* Not all importance methods are the same:

  * Tree-based importance (fast, built-in).
  * Coefficients in linear models (logistic regression).
  * More advanced methods (e.g., SHAP – later in the course / other projects).
* Interpretation must always be combined with **domain knowledge**:

  * Does it make sense that pitch and HNR matter for trust?
  * Are any results surprising or suspicious?
* Even with feature importance, **correlation ≠ causation**:

  * A feature being important does not prove it “causes” trust.

---

#### 3. Real-World Examples or Stories

1. **Teacher grading a project**
   When grading, you may have a “mental model” of what matters: clarity, depth, originality. Feature importance is like listing, “Clarity: 40%, Depth: 35%, Originality: 25%.”

2. **Car safety rating**
   Suppose a model predicts: “Is this car safe?” Features might include number of airbags, braking distance, stability control, etc. Feature importance might show that braking distance and airbag count matter most.

3. **Food delivery time**
   Predicting delivery time: distance, time of day, driver experience, traffic conditions. Feature importance reveals which factors mostly drive delay.

4. **Trust in voices**
   The paper suggests that certain voice characteristics (pitch, voice quality) are strong cues of trust. By inspecting feature importance, we check whether the model “agrees” with psychological theories and human perception.

---

#### 4. Terminology Explained

* **Feature importance** – A score for each input feature indicating how much it contributes to model decisions.
* **Random Forest feature_importances_** – A built-in attribute of RandomForest models in scikit-learn that gives an importance score per feature.
* **Impurity** – A measure of how mixed classes are in a decision tree node (e.g., Gini impurity). Lower impurity = more “pure” node.
* **Impurity decrease** – How much a split on a feature reduces impurity; more reduction → more important.
* **Global importance** – Average importance of features over all predictions and all trees (not per individual instance).
* **Coefficient (in logistic regression)** – Numbers multiplying each feature; large positive or negative coefficients indicate strong influence.
* **Domain knowledge** – Expert understanding of the problem (here, how humans change their voices when being trustworthy).

---

#### 5. How It Works – Random Forest Feature Importances

We start with the same kind of dataset you’ve seen:

```text
duration_seconds | mean_f0_hz | sd_f0_hz | hnr_db | shimmer_db | cpp_db | intent_label
--------------------------------------------------------------------------------------
1.65             | 190.3      | 22.1     | 11.0   | 0.34       | 15.2   | 1
1.78             | 175.2      | 18.3     |  9.6   | 0.43       | 13.1   | 0
...
```

**Step 1 – Train a Random Forest (like in Lesson 5)**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feature_columns = ["duration_seconds", "mean_f0_hz", "sd_f0_hz",
                   "hnr_db", "shimmer_db", "cpp_db"]
X = data[feature_columns]
y = data["intent_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
```

**Step 2 – Get feature importances**

```python
importances = rf.feature_importances_
for name, score in zip(feature_columns, importances):
    print(name, ":", score)
```

* `importances` is an array, e.g.:

```text
duration_seconds : 0.07
mean_f0_hz       : 0.28
sd_f0_hz         : 0.14
hnr_db           : 0.22
shimmer_db       : 0.09
cpp_db           : 0.20
```

**Interpretation:**

* `mean_f0_hz` and `hnr_db` might be most important.
* This matches the idea that **pitch** and **voice clarity** are strong trust cues.

**Step 3 – Sort and visualize**

```python
import numpy as np
import matplotlib.pyplot as plt

sorted_indices = np.argsort(importances)[::-1]
sorted_names = [feature_columns[i] for i in sorted_indices]
sorted_vals = importances[sorted_indices]

plt.figure()
plt.bar(range(len(sorted_names)), sorted_vals)
plt.xticks(range(len(sorted_names)), sorted_names, rotation=45)
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
```

**Step 4 – Connect to the paper**

* If the paper reports that certain acoustic features are particularly predictive (e.g., increased F0, increased HNR, changed speaking rate), we can see whether our feature importance ranking aligns with those findings.
* This is a key step in “reproducing and understanding” the paper, not just copying code.

---

#### 6. Practice Exercises

**Exercise 1 – Reading Importances**

Suppose the model prints:

```text
duration_seconds : 0.08
mean_f0_hz       : 0.30
sd_f0_hz         : 0.10
hnr_db           : 0.25
shimmer_db       : 0.07
cpp_db           : 0.20
```

a) Which feature is most important?
b) Which feature is least important?

---

**Exercise 2 – Ranking Features**

Based on the importances above, list the features from **most important to least important**.

---

**Exercise 3 – Interpretation**

Using the same importances, write 2–3 sentences explaining what the model thinks are the most important cues for trustworthy vs neutral speech.

---

**Exercise 4 – Importance vs Performance**

True or False: “If one feature has high importance, then the model’s accuracy must also be very high.”

Explain your answer briefly.

---

**Exercise 5 – Domain Knowledge Check**

Suppose the model says `duration_seconds` is by far the most important feature, much higher than pitch or HNR.

a) What potential issues might this indicate?
b) What follow-up steps could you take?

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

a) Most important: **`mean_f0_hz`** (0.30).
b) Least important: **`shimmer_db`** (0.07) (tied with `duration_seconds`, but shimmer is slightly lower in this example).

---

**Exercise 2 – Answer**

From most to least important:

1. `mean_f0_hz` (0.30)
2. `hnr_db` (0.25)
3. `cpp_db` (0.20)
4. `sd_f0_hz` (0.10)
5. `duration_seconds` (0.08)
6. `shimmer_db` (0.07)

---

**Exercise 3 – Sample Answer**

The model relies most on `mean_f0_hz` (average pitch), followed by `hnr_db` (voice clarity) and `cpp_db` (a voice quality measure). This suggests that pitch and voice quality are strong indicators of whether an utterance is trustworthy or neutral in this dataset. Duration and shimmer appear less important, meaning the model doesn’t depend as heavily on how long the utterance is or small amplitude variations.

---

**Exercise 4 – Answer**

False. A feature can have high importance **relative to other features**, but the overall model accuracy might still be low if the task is hard, the dataset is noisy, or there isn’t enough data. Feature importance tells us which features the model uses most, not how good the model is in absolute terms.

---

**Exercise 5 – Sample Answer**

a) If `duration_seconds` is dominating importance, it might mean:

* The dataset is biased (e.g., trustworthy utterances always longer).
* The model is “cheating” by exploiting some artifact of how data was recorded.
* Other features weren’t extracted correctly.

b) Follow-up steps:

* Check distributions of duration by label; see if they’re unrealistically separated.
* Revisit feature extraction and data pre-processing.
* Retrain with duration removed or penalized to see if results become more realistic.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Are feature importances always trustworthy?
   **A:** They’re helpful but not perfect. They depend on the model type and how features are correlated. Use them as clues, not absolute truth.

2. **Q:** Do feature importances tell us causation?
   **A:** No. They show which features the model relies on, not which features *cause* trust.

3. **Q:** Why might a feature with small variance still be important?
   **A:** If small changes in that feature strongly affect the label, the model may learn to rely on it even if its variance is small.

4. **Q:** Can we compare importances across different models?
   **A:** Roughly, yes, but be careful: different models calculate importance in different ways, so comparisons are not exact.

5. **Q:** What if two features are highly correlated?
   **A:** Their importances can split or one can “steal” importance from the other. Interpretation gets trickier.

6. **Q:** How do logistic regression coefficients relate to importance?
   **A:** Larger absolute coefficients usually mean higher importance, but they depend on feature scaling and direction (positive/negative).

7. **Q:** Why do we still need domain knowledge if we have importances?
   **A:** Domain knowledge helps you judge whether the results make sense and spot weird artifacts or biases.

8. **Q:** What is the difference between global and local explanation?
   **A:** Global explains importance across the whole dataset; local explains why the model made a specific prediction for one example.

9. **Q:** Could we use SHAP values instead of Random Forest importances?
   **A:** Yes. SHAP is more advanced and can provide local explanations; we can introduce that later once basics are solid.

10. **Q:** Why does the paper care about interpretation?
    **A:** Because we want to understand *how* humans signal trustworthy intent in their voices, not just build a black-box classifier.

---

#### 9. Quiz (10 Questions + Answers + Explanations)

**Q1.** Feature importance in a Random Forest tells us:
a) Exact causal effects
b) How often a feature is missing
c) How much a feature contributes to the model’s decisions
d) The training time of the model

**Answer:** c)
**Explanation:** Importances measure contribution to splits and impurity reduction, not causality or runtime.

---

**Q2.** If `mean_f0_hz` has importance 0.35 and `hnr_db` has 0.25, what can we say?
**Answer:** `mean_f0_hz` is more influential than `hnr_db` in the model’s decisions, but both are relatively important.

---

**Q3.** True or False: “Feature importances always sum to 1 in scikit-learn’s RandomForest.”

**Answer:** True.
**Explanation:** They are normalized to sum to 1.

---

**Q4.** Which of the following is *not* a voice-related feature?
a) `mean_f0_hz`
b) `hnr_db`
c) `height_cm`
d) `shimmer_db`

**Answer:** c)
**Explanation:** The others are acoustic; height is not.

---

**Q5.** If all features have similar importance values, what might that mean?
**Answer:** The model is using all features somewhat evenly; no single feature dominates, or features may be correlated in a way that shares importance.

---

**Q6.** In this trust detection scenario, which features are most likely to be important according to domain knowledge?
a) Random ID numbers
b) Pitch and voice quality measures
c) File name length
d) Recording index

**Answer:** b)
**Explanation:** Research suggests pitch and voice quality carry trust cues.

---

**Q7.** If a feature importance is 0, what does that usually mean?
**Answer:** The model did not use that feature in any split and found it irrelevant for predictions.

---

**Q8.** What is a good next step after seeing feature importances?
a) Immediately delete all low-importance features
b) Check if the ranking matches domain expectations and inspect suspicious results
c) Ignore them
d) Only keep the most important feature

**Answer:** b)
**Explanation:** We use importances for insight, not automatic pruning without thinking.

---

**Q9.** True or False: “High feature importance guarantees high accuracy on new speakers.”

**Answer:** False.
**Explanation:** Importance describes relative influence; generalization depends on many factors and must be tested.

---

**Q10.** Why is feature importance helpful when reproducing a research paper?
**Answer:** It lets us check whether our model relies on the same kinds of cues the authors describe, increasing confidence that we’re replicating their findings, not just matching numbers.

---

#### 10. Mini Practice Project – Feature Importance for Trust (Downloadable .zip)

**Mini Project Title:** *Inspecting Acoustic Feature Importance in a Trust Classifier*

I’ve prepared a mini project folder with:

* `synthetic_trust_feature_importance_dataset.csv`

  * Rows = utterances
  * Columns:

    * `duration_seconds`
    * `mean_f0_hz`
    * `sd_f0_hz`
    * `hnr_db`
    * `shimmer_db`
    * `cpp_db`
    * `intent_label`

* `feature_importance_trust.py`

  * Fully commented script that:

    * Loads the dataset
    * Splits into train/test
    * Trains a Random Forest
    * Computes test accuracy
    * Extracts `feature_importances_`
    * Prints raw and sorted importances
    * Plots a bar chart of feature importances

* `README.txt` explaining how to run everything.

**Student instructions:**

1. Unzip the file.

2. Open a terminal in `lesson7_feature_importance`.

3. Install required packages (once):

   ```bash
   pip install pandas scikit-learn numpy matplotlib
   ```

4. Run:

   ```bash
   python feature_importance_trust.py
   ```

5. Look at:

   * Test accuracy.
   * Printed importance values.
   * Bar chart of sorted importances.

6. Answer:

   * Which feature is most important?
   * Does that match your intuition about trustworthy vs neutral speech?

**Optional extension:**

* Remove one high-importance feature (e.g., drop `mean_f0_hz`) and re-train.
* Observe how the accuracy and remaining importances change.

---

#### 11. References

* scikit-learn documentation: RandomForestClassifier and `feature_importances_`.
* Intro resources on:

  * “Random Forest feature importance”
  * “Interpreting machine learning models.”
* Related conceptual follow-ups:

  * “Model-agnostic interpretation methods (e.g., permutation importance, SHAP).”

---

#### 12. Additional Information / Teacher Tips

* **Link to SHAP later**
  This lesson sets up the idea of “which features matter?” Later, you can introduce SHAP to explain **per-utterance** contributions, tying back to your SHAP/SHapley materials from other projects.

* **Connect to psychology**
  Encourage the student to think:

  > “If the model says pitch and HNR are important, do human listeners also report these cues when judging trust?”

* **Encourage curiosity & skepticism**
  Ask students to find at least one result that **surprises** them and write a short paragraph exploring why it might be happening.
