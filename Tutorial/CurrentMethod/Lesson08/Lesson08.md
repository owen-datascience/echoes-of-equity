### Lesson 8 – Human Trust Ratings & Creating Labels from Crowd Data

---

#### 1. Lesson Summary

In this lesson, the student learns how to go from **raw human trust ratings** (many people rating many audio clips) to **usable labels** for machine learning.

We’ll focus on:

* How rating studies in the paper work:

  * Many raters listen to each voice clip and give a **trust rating** (e.g., 1–7 scale).
* How to organize these data in “long” format: one row = one rater’s judgment of one utterance.
* How to compute **utterance-level statistics**:

  * Mean rating
  * Standard deviation (how much raters disagree)
  * Number of ratings per clip
* How to convert continuous ratings into **binary labels** (e.g., “trustworthy” vs “neutral/low trust”) using a threshold.
* How to visualize rating distributions and disagreements.

This mirrors what the paper does conceptually: they gather human trust ratings and then create labels and summary statistics for each clip.

---

#### 2. Key Points

* The dataset includes **human judgments**, not just acoustic features.

* Each utterance is rated by **multiple raters**, so we have many rows per utterance.

* We usually store ratings in **long format**:

  | utterance_id | rater_id | trust_rating |
  | ------------ | -------- | ------------ |
  | U001         | R01      | 6            |
  | U001         | R02      | 5            |
  | U001         | R03      | 7            |
  | ...          | ...      | ...          |

* To use these ratings in models, we typically compute **utterance-level aggregates**:

  * Mean rating (how trustworthy this clip sounds on average).
  * Standard deviation (how much raters disagree).
  * Count of ratings.

* To train a **binary classifier**, we often convert mean ratings into 0/1 labels:

  * Example: mean ≥ 4.0 (on 1–7 scale) → label = 1 (trustworthy).
  * mean < 4.0 → label = 0 (neutral/low trust).

* We can visualize:

  * Histogram of mean ratings → how many clips are trusted or not.
  * Scatter of mean vs standard deviation → which clips are controversial (high disagreement).

* This step is crucial to reproducing the paper: you must **understand how labels are constructed** from human ratings.

---

#### 3. Real-World Examples or Stories

1. **Movie ratings**
   Many people rate a movie 1–5 stars. We take the **average rating** to say “this movie is 4.2 stars”. In ML, we could then define “good movie” = average ≥ 3.5.

2. **Teacher evaluations**
   Students rate a teacher from 1–7. The school might use the **mean rating** to summarize teaching quality and look at **disagreement** (some students strongly like, others dislike).

3. **Restaurant review scores**
   Thousands of people give 1–5 star ratings. Sites display the mean rating and sometimes highlight restaurants with both high mean and many ratings.

4. **Trust in voices**
   In the paper, listeners rate “how trustworthy does this voice sound?”
   We need to combine many ratings per utterance into a single score or label to train ML models.

---

#### 4. Terminology Explained

* **Long format** – Each row is one rater’s rating for one item (here, one voice clip).
* **Aggregated ratings** – Combining many ratings for the same item into summary numbers (mean, std, count).
* **Mean rating** – Average of all ratings for one utterance.
* **Standard deviation (std)** – A measure of how spread out ratings are; higher std = more disagreement.
* **Thresholding** – Choosing a cutoff (e.g., mean >= 4.0) to turn continuous ratings into categories.
* **Binary label** – A label with only two values (e.g., 0 = neutral/low trust, 1 = trustworthy).
* **Inter-rater agreement** – How much raters agree with each other (we get a sense of this from std).

---

#### 5. How It Works – From Ratings to Labels

We start with a ratings table like this (long format):

```text
utterance_id | rater_id | trust_rating
--------------------------------------
U001         | R01      | 6
U001         | R02      | 5
U001         | R03      | 7
U002         | R01      | 3
U002         | R02      | 4
...
```

**Step 1 – Load and inspect**

```python
import pandas as pd

ratings = pd.read_csv("synthetic_trust_ratings_long.csv")
print(ratings.head())
print(ratings.shape)  # (rows, columns)
print("Unique utterances:", ratings["utterance_id"].nunique())
print("Unique raters:", ratings["rater_id"].nunique())
```

**Step 2 – Group by utterance**

We want statistics per utterance, so we group:

```python
grouped = ratings.groupby("utterance_id")

aggregated = grouped["trust_rating"].agg(
    mean_rating="mean",
    std_rating="std",
    n_ratings="count"
).reset_index()

print(aggregated.head())
```

Now each row is **one utterance**:

```text
utterance_id | mean_rating | std_rating | n_ratings
---------------------------------------------------
U001         | 6.0         | 1.0        | 12
U002         | 3.5         | 1.3        | 12
...
```

**Step 3 – Create a binary label**

Define a threshold, e.g.:

```python
threshold = 4.0
aggregated["intent_label"] = (aggregated["mean_rating"] >= threshold).astype(int)
print(aggregated["intent_label"].value_counts())
```

Now we have:

* `intent_label = 1` for clips rated as trustworthy on average.
* `intent_label = 0` for clips rated less trustworthy.

We can now join this `aggregated` table with our **acoustic features** table (by `utterance_id`) to train models.

**Step 4 – Visualize distributions**

Histogram of mean ratings:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.hist(aggregated["mean_rating"], bins=10)
plt.title("Histogram of mean trust ratings per utterance")
plt.xlabel("Mean rating (1–7)")
plt.ylabel("Number of utterances")
plt.tight_layout()
plt.show()
```

Scatter of mean vs disagreement:

```python
plt.figure()
plt.scatter(aggregated["mean_rating"], aggregated["std_rating"])
plt.title("Mean rating vs rating disagreement (std)")
plt.xlabel("Mean rating")
plt.ylabel("Standard deviation of ratings")
plt.tight_layout()
plt.show()
```

* High mean, low std → almost everyone agrees the voice sounds trustworthy.
* Low mean, low std → everyone agrees it doesn’t sound trustworthy.
* Mid mean, high std → raters disagree.

**Step 5 – Save utterance-level table for modeling**

```python
aggregated.to_csv("trust_utterance_level_labels.csv", index=False)
```

This file will be used in later lessons to create **full model pipelines** that combine human labels and acoustic features.

---

#### 6. Practice Exercises

**Exercise 1 – Interpreting Mean Rating**

An utterance has ratings: [5, 6, 6, 7, 5, 6].
a) What is the mean rating?
b) Would you label it as “trustworthy” if the threshold is 4.0?

---

**Exercise 2 – Interpreting Disagreement**

Two utterances have these ratings:

* Utterance A: [4, 4, 4, 4, 4, 4]
* Utterance B: [1, 7, 1, 7, 1, 7]

a) Which utterance has higher mean rating?
b) Which has higher standard deviation?
c) Which one shows more rater disagreement?

---

**Exercise 3 – Counting Labels**

Suppose we have this aggregated table:

```text
utterance_id | mean_rating
---------------------------
U001         | 5.1
U002         | 3.8
U003         | 4.2
U004         | 2.9
U005         | 4.0
```

With threshold 4.0 (>= is label 1), what is `intent_label` for each utterance?

---

**Exercise 4 – Threshold Experiment**

For the table above, try thresholds 3.5 and 4.5.
How many utterances would be labeled as 1 (trustworthy) in each case?

---

**Exercise 5 – Why Aggregation?**

Explain in 2–3 sentences why we aggregate multiple raters into one mean rating per utterance, instead of using raw rater-level rows directly in training.

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

Ratings: [5, 6, 6, 7, 5, 6]

* Mean = (5 + 6 + 6 + 7 + 5 + 6) / 6 = 35 / 6 ≈ **5.83**.
* With threshold 4.0, mean 5.83 ≥ 4.0 → label as **trustworthy (1)**.

---

**Exercise 2 – Answer**

Utterance A: [4, 4, 4, 4, 4, 4]

* Mean = 4.0, std = 0 (no spread).

Utterance B: [1, 7, 1, 7, 1, 7]

* Mean = (1+7+1+7+1+7)/6 = 24/6 = 4.0, but std is high (ratings alternate between extremes).

a) Both have the **same mean** (4.0).
b) **Utterance B** has higher standard deviation.
c) **Utterance B** shows more rater disagreement.

---

**Exercise 3 – Answer**

Threshold 4.0, label = 1 if mean_rating ≥ 4.0, else 0.

* U001: 5.1 → 1
* U002: 3.8 → 0
* U003: 4.2 → 1
* U004: 2.9 → 0
* U005: 4.0 → 1

---

**Exercise 4 – Answer**

Threshold 3.5 (≥ 3.5 → 1):

* U001: 5.1 → 1
* U002: 3.8 → 1
* U003: 4.2 → 1
* U004: 2.9 → 0
* U005: 4.0 → 1

→ 4 utterances labeled 1.

Threshold 4.5 (≥ 4.5 → 1):

* U001: 5.1 → 1
* U002: 3.8 → 0
* U003: 4.2 → 0
* U004: 2.9 → 0
* U005: 4.0 → 0

→ 1 utterance labeled 1.

---

**Exercise 5 – Sample Answer**

We aggregate multiple raters to get a **more stable and reliable** estimate of how trustworthy an utterance sounds. Individual raters may be noisy or have strong personal biases, but averaging over many of them smooths out random variation. Using one mean rating or label per utterance also makes it simpler to join with acoustic features and train a standard classifier.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why not use just one rater per clip?
   **A:** With one rater, results are very noisy and depend strongly on that person’s biases. Multiple raters give a more reliable estimate.

2. **Q:** Why use a 1–7 scale instead of 0/1 directly?
   **A:** A 1–7 scale captures more nuance (slightly trustworthy vs very trustworthy). We can always convert it to 0/1 later.

3. **Q:** How do we choose the threshold (like 4.0)?
   **A:** Often we choose the neutral midpoint of the scale, or we experiment with thresholds that balance the number of 0 and 1 labels.

4. **Q:** Why compute standard deviation of ratings?
   **A:** It tells us how much raters disagree. High disagreement means the clip is “controversial” or ambiguous.

5. **Q:** Could we keep the ratings as continuous targets instead of turning them into 0/1?
   **A:** Yes. Then we’d do **regression** instead of classification. The paper may analyze ratings both ways.

6. **Q:** What if some utterances have fewer ratings than others?
   **A:** We’d check `n_ratings` and might treat utterances with very few ratings more carefully, or exclude them.

7. **Q:** Can we weight raters differently?
   **A:** In more advanced setups, yes (e.g., raters with high reliability might get more weight), but we usually start with simple averages.

8. **Q:** Does the paper use exactly the same threshold?
   **A:** Not necessarily; different papers define “trustworthy” in slightly different ways. The important idea is understanding the process.

9. **Q:** Is it okay that mean and std are real numbers while ratings are integers?
   **A:** Yes. Mean and std summarize the distribution; they don’t need to be integers.

10. **Q:** How does this step help with reproducing the paper?
    **A:** The paper’s labels and summary statistics come from human ratings. Reproducing their analysis means replicating how they aggregated ratings and defined trust labels.

---

#### 9. Quiz (10 Questions + Answers + Explanations)

**Q1.** In long format, each row corresponds to:
a) One utterance
b) One rater
c) One [utterance, rater] rating
d) One model prediction

**Answer:** c)
**Explanation:** Long format stores one rating per row, identified by utterance_id and rater_id.

---

**Q2.** What is the main reason to compute the mean rating per utterance?
**Answer:** To summarize many individual ratings into a single value that represents how trustworthy the utterance sounds on average.

---

**Q3.** Standard deviation of ratings measures:
a) The average rating
b) The total number of ratings
c) How much raters disagree
d) Whether ratings are correct

**Answer:** c)
**Explanation:** Std quantifies spread; higher std = more disagreement.

---

**Q4.** True or False: “If all raters give exactly the same rating to an utterance, std = 0.”

**Answer:** True.
**Explanation:** No variation → zero standard deviation.

---

**Q5.** If we set a very high threshold for trust (e.g., 5.5 on a 1–7 scale), how will that affect the number of positive labels?
**Answer:** It will reduce the number of utterances labeled as trustworthy (positive), making the dataset more imbalanced.

---

**Q6.** Which of the following is *not* a reason to aggregate ratings?
a) Reduce noise
b) Summarize many ratings into one value
c) Make it easier to combine with acoustic features
d) Guarantee perfect accuracy

**Answer:** d)
**Explanation:** Aggregation helps, but doesn’t guarantee perfection.

---

**Q7.** If an utterance has mean_rating = 3.2 and std_rating = 0.2, what does that suggest?
**Answer:** Raters mostly agree that the utterance is slightly below neutral in trust (low disagreement around a low-ish mean).

---

**Q8.** When we save `trust_utterance_level_labels.csv`, what is it mainly used for?
**Answer:** As a label file to be joined with acoustic features so we can train and evaluate ML models.

---

**Q9.** True or False: “The choice of threshold does not affect the model at all.”

**Answer:** False.
**Explanation:** Threshold changes how labels are assigned, which changes the classification task and class balance.

---

**Q10.** In the context of this paper, why is it important to look at rating distributions, not just labels?
**Answer:** Because distributions show how strong and how consistent trust judgments are. Two utterances with the same label may have very different levels of agreement and confidence.

---

#### 10. Mini Practice Project – Aggregating Human Trust Ratings

**Mini Project Title:** *From Human Ratings to Trust Labels*

I’ve prepared a mini project with:

* `synthetic_trust_ratings_long.csv`

  * Columns:

    * `utterance_id` – U001, U002, …
    * `rater_id` – R01, R02, …
    * `trust_rating` – integer 1–7

* `aggregate_trust_ratings.py`

  * Fully commented script that:

    * Loads the ratings table
    * Computes number of utterances and raters
    * Aggregates per utterance: `mean_rating`, `std_rating`, `n_ratings`
    * Creates `intent_label` using threshold 4.0
    * Plots:

      * Histogram of mean ratings
      * Scatter of mean vs std
    * Saves `trust_utterance_level_labels.csv` for modeling.

* `README.txt` with clear instructions.

**Suggested student steps:**

1. Unzip the folder.

2. Install packages if needed:

   ```bash
   pip install pandas numpy matplotlib
   ```

3. Run:

   ```bash
   python aggregate_trust_ratings.py
   ```

4. Examine:

   * The aggregated table.
   * Label counts.
   * Plots of mean ratings and disagreement.

5. Experiment:

   * Change the threshold (e.g., 3.5, 4.5) and see how label counts change.
   * Think about which threshold seems more reasonable.

---

#### 11. References

* General reading on:

  * “Likert scales and rating aggregation.”
  * “Inter-rater reliability and agreement.”
* Example tutorials that discuss:

  * Converting human ratings to labels for ML.
  * Using `groupby` in pandas for aggregation.

---

#### 12. Additional Information / Teacher Tips

* **Tie back to the paper**
  Emphasize that the paper’s labels and analysis start from **exactly this type of ratings data**. Understanding this step gives the student a deeper appreciation of the dataset.

* **Connect to statistics gently**
  You can briefly mention more advanced reliability metrics (like Cronbach’s alpha or ICC) as future topics, but keep this lesson focused on intuitive measures (mean and std).

* **Bridge to next lessons**
  Next, we can:

  * Join utterance-level labels with acoustic features.
  * Recreate something close to the paper’s **full pipeline**: LOSO, logistic regression, random forest, metrics, and comparison to human ratings.
