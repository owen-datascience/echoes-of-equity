### Lesson 3 – Understanding the Trust Dataset: Speakers, Sentences & Labels

---

#### 1. Lesson Summary

In this lesson, the student will learn **how the real “trustworthy intent” dataset is organized**:

* What counts as **one data point** (an utterance).
* How **speakers**, **sentences**, and **intents** (neutral vs trustworthy) are stored.
* What **metadata** is included (age group, sex, ethnicity).
* How this structure supports **fair machine learning**, where we can check performance for different demographic groups.

We’ll also connect this to **practical skills**: reading a CSV with pandas, exploring rows/columns, counting speakers, and checking class balance. The student will practice this with a small synthetic dataset that mimics the structure of the real one.

---

#### 2. Key Points

* The real dataset contains **1,152 utterances from 96 speakers**. Each utterance is one short spoken sentence.
* Speakers come from **three ethnic backgrounds** (white, Black, South Asian), **two age groups** (younger 18–45, older 60+), and both **sexes**.
* Each speaker recorded sentences in **two intents**:

  * Neutral (no special intent)
  * Trustworthy intent (trying to sound trustworthy)
* The data table is organized with:

  * **One row per utterance** (sentence spoken once)
  * Columns for **speaker metadata** (ID, age group, sex, ethnicity)
  * Columns for **experimental condition** (neutral vs trustworthy)
  * Columns for **acoustic features** (pitch, HNR, etc.)
* This structure lets us:

  * Count how many examples per class (neutral/trustworthy)
  * Check how examples are distributed across demographics
  * Apply **cross-validation** where we hold out speakers (not just random rows).
* Understanding the dataset structure is essential before training any model.

---

#### 3. Real-World Examples or Stories

1. **Class roster analogy**
   Imagine a spreadsheet listing all your classmates. Each row is one **student**; columns: name, age, grade, club membership. Here the dataset is similar, but each row is one **utterance**: who spoke it, how, and in what condition.

2. **Survey data**
   When you fill out a survey (age, gender, answers to questions), the results end up in a table. In this dataset, each utterance is like one survey response, with the **voice features** and **intent label** as answers.

3. **Sports stats**
   In a basketball stats table, each row might be one game: player, opponent, points, rebounds, etc. For this project, each row is one spoken sentence: speaker, intent, duration, pitch, etc.

4. **Fairness in grading**
   If a teacher only looked at a few students’ tests, they might accidentally bias the results. Similarly, if our dataset had mostly young white male speakers, our model might not generalize to others. That’s why the paper carefully balances demographics.

---

#### 4. Terminology Explained

* **Observation / sample / example** – One row in the dataset; here, one utterance.
* **Metadata** – Extra information about each example that is not the main target (e.g., speaker ID, age group).
* **Label** – The target value we want to predict (here: neutral vs trustworthy intent).
* **Class** – One category of the label (neutral or trustworthy).
* **Class balance** – How many examples belong to each class. A balanced dataset has similar counts.
* **Speaker ID** – A code (e.g., S01, S02) identifying each speaker.
* **Within-subject design** – Each speaker appears in both conditions (neutral and trustworthy); we compare each person to themselves.
* **Train / test split** – Dividing data into a part for training the model and a part for evaluating it.
* **Speaker-independent evaluation** – Making sure that speakers in the training set are **not** in the test set, to test generalization to new people.

---

#### 5. How It Works – Dataset Structure in Practice

Let’s imagine a simplified version of the real trust dataset table.

**One row per utterance**
Example row:

| speaker_id | age_group | sex  | ethnicity   | sentence_id | intent_label | duration | mean_F0 |  HNR | … |
| ---------: | --------- | ---- | ----------- | ----------- | -----------: | -------: | ------: | ---: | - |
|        S12 | younger   | male | south_asian | 3           |            1 |     1.65 |   193.4 | 11.0 | … |

* `speaker_id` – which person spoke
* `age_group` – younger or older
* `sex` – male or female
* `ethnicity` – white / Black / South Asian
* `sentence_id` – which sentence prompt they read (like “You can count on me”)
* `intent_label` – 0 = neutral, 1 = trustworthy
* Acoustic features follow: `duration`, `mean_F0`, `HNR`, etc.

**Within each speaker:**

* They read multiple sentences.
* Each sentence is recorded twice: **neutral** and **trustworthy intent**.
* That means for each speaker, there are multiple rows with `intent_label = 0` and multiple with `intent_label = 1`.

**Why this structure is important:**

* We can **group by `speaker_id`** to see how many utterances each speaker has.
* We can check that **each speaker appears in both classes**, which is crucial for the experimental design.
* We can group by demographic columns (e.g., `ethnicity`) to see if the dataset is balanced across groups.
* For fair evaluation, we can perform **leave-one-speaker-out cross-validation**:

  * Train on 95 speakers, test on the 1 left-out speaker
  * Repeat for every speaker
    This tests whether the model generalizes to **new voices**.

In this lesson’s mini project, the student will work with a small synthetic dataset that has the same **logical structure**, so they can practice with pandas before touching the full dataset.

---

#### 6. Practice Exercises

**Exercise 1 – Identify the Observation**

In the trust dataset, what is considered **one observation** (one row)?
a) One speaker
b) One sentence text (like “You can count on me”)
c) One utterance (a sentence spoken once with a particular intent)
d) One feature like pitch

---

**Exercise 2 – Matching Columns**

Match each concept to the most likely column name:

1. Whether the speaker tried to sound trustworthy
2. Which demographic group the speaker belongs to (e.g., 18–45 or 60+)
3. Which person spoke the sentence
4. The average pitch of that utterance

Columns: `speaker_id`, `age_group`, `intent_label`, `mean_f0_hz`

---

**Exercise 3 – Why Metadata Matters**

Explain in 3–4 sentences why it is important to store metadata like `age_group`, `sex`, and `ethnicity` when building a dataset for trust in voices.

---

**Exercise 4 – Counting Examples**

Suppose you load the dataset and compute:

```python
data["intent_label"].value_counts()
```

The output is:

```text
0    580
1    572
Name: intent_label, dtype: int64
```

a) Which label is more frequent?
b) Is the dataset roughly balanced? Why or why not?

---

**Exercise 5 – Speaker-Independent Splits**

Why is it a bad idea to randomly split rows for train/test if some rows from the **same speaker** end up in both sets? Explain in 2–3 sentences.

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

Correct answer: **c) One utterance**
Each row in the dataset represents one specific recording of a sentence spoken once under a particular intent.

---

**Exercise 2 – Answer**

1 → `intent_label`
2 → `age_group`
3 → `speaker_id`
4 → `mean_f0_hz`

---

**Exercise 3 – Sample Answer**

Metadata like age group, sex, and ethnicity tells us **who** the voices belong to. This helps ensure that the dataset is not dominated by only one group, which could make the model biased. It also allows researchers to check whether the model performs equally well for different groups, which is important for fairness and generalization in real-world applications.

---

**Exercise 4 – Answer**

a) Label **0** (neutral) is slightly more frequent (580 vs 572).
b) Yes, it’s roughly balanced because the numbers are very close. There is no huge difference between the counts, so the model will see a similar number of examples from each class.

---

**Exercise 5 – Sample Answer**

If rows from the same speaker appear in both train and test sets, the model might “cheat” by learning **speaker-specific patterns** instead of general patterns about trust. It could perform very well on the test set simply because it recognizes the speaker’s voice, not because it truly understands trustworthy intent. That’s why we prefer speaker-independent splits like leave-one-speaker-out.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why not store one row per speaker instead of one row per utterance?
   **A:** Because each speaker produces multiple utterances under different conditions. We need separate rows to capture each specific recording and its features.

2. **Q:** Can a dataset have multiple labels?
   **A:** Yes. For example, you could have `intent_label` (neutral vs trustworthy) and `emotion_label` (happy, sad, etc.). In this study, the main label is intent.

3. **Q:** What is the difference between `speaker_id` and `sentence_id`?
   **A:** `speaker_id` tells you **who** spoke; `sentence_id` tells you **which sentence** they were reading.

4. **Q:** Why is class balance important?
   **A:** If one class is much larger, a model might just guess that class all the time and still get high accuracy, which is misleading.

5. **Q:** What is “within-subject” design?
   **A:** It means each participant (speaker) experiences all conditions (neutral and trustworthy), so we can compare how each person changes their voice between conditions.

6. **Q:** Could we treat each speaker as a separate dataset?
   **A:** You could analyze each speaker individually, but for machine learning you usually want to combine them to learn general patterns while still keeping track of who is who.

7. **Q:** Why are demographic columns not used directly as features for trust classification?
   **A:** The goal is to detect trust from **how the voice sounds**, not from who the person is. Demographic info is mainly for analyzing fairness and dataset diversity, not for prediction.

8. **Q:** How do we represent text content in this dataset?
   **A:** In this paper, the focus is on **acoustic features**, not text. The sentence ID indicates which prompt was used, but we mostly ignore the actual words.

9. **Q:** Is it okay if some speakers have a few more utterances than others?
   **A:** Slight differences are okay, but very unbalanced speakers could bias the model. The paper’s design aims for relatively consistent counts per speaker.

10. **Q:** Can we add more metadata later?
    **A:** Yes, as long as you can link it to existing rows via `speaker_id` or a file name. For example, you could later add hearing test results or personality measures if available.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** In this dataset, what does each row represent?
a) One speaker
b) One acoustic feature
c) One utterance (sentence spoken once)
d) One demographic group

**Answer:** c) One utterance.

---

**Q2.** Which column is most likely the **label** for our classification task?
a) `speaker_id`
b) `intent_label`
c) `age_group`
d) `mean_f0_hz`

**Answer:** b) `intent_label`.

---

**Q3.** What type of information is `age_group`?
a) Feature used for prediction
b) Label
c) Metadata
d) Noise

**Answer:** c) Metadata (though it *could* be used as a feature for analysis, its main role is contextual info).

---

**Q4.** True or False: The dataset includes both neutral and trustworthy recordings for each speaker.

**Answer:** True (that’s part of the within-subject design).

---

**Q5.** Why do we care about how many examples each class has?
**Answer:** Because a highly imbalanced dataset can lead to biased models and misleading accuracy; balanced classes make evaluation more meaningful.

---

**Q6.** What does `speaker_id` help us do?
a) Measure pitch
b) Identify which group a sample belongs to
c) Track which utterances belong to the same speaker
d) Compute duration

**Answer:** c) Track which utterances belong to the same speaker.

---

**Q7.** Which of the following is **not** metadata?
a) `speaker_id`
b) `age_group`
c) `intent_label`
d) `ethnicity`

**Answer:** c) `intent_label` – that’s the target label.

---

**Q8.** Why is it good that the dataset includes different ethnicities and age groups?
**Answer:** It reduces bias and makes the results more generalizable across diverse speakers.

---

**Q9.** True or False: When evaluating the model, it’s better to keep all utterances from a single speaker either in train or test, but not both.

**Answer:** True – this is speaker-independent evaluation.

---

**Q10.** If we see that one demographic group has very few examples, what might that indicate?
**Answer:** The dataset may be unbalanced across demographics, and we should be careful about drawing conclusions for that group; the model may not perform well for them.

---

#### 10. Mini Practice Project – Synthetic Trust Dataset Structure (with .zip)

**Mini Project Title:** *Exploring a Synthetic Trust Dataset Structure with Pandas*

Goal: Let the student practice **dataset exploration** on a small synthetic table that mimics the structure of the full trust dataset.

I’ve created a project folder that includes:

* `synthetic_trust_dataset_structure.csv`

  * Columns:

    * `speaker_id` – e.g., S01, S02, …
    * `age_group` – `"younger"` or `"older"`
    * `sex` – `"female"` or `"male"`
    * `ethnicity` – `"white"`, `"black"`, `"south_asian"`
    * `sentence_id` – integer ID of the sentence per speaker
    * `intent_label` – 0 = neutral, 1 = trustworthy
    * `duration_seconds` – utterance duration
    * `mean_f0_hz` – mean pitch

* `explore_trust_dataset_structure.py`

  * Fully commented, doing:

    * Load CSV with pandas
    * Print head, shape, column names
    * Count utterances per speaker
    * Check overall class distribution (`intent_label`)
    * Group by `speaker_id` and `intent_label` to verify each speaker has both intents
    * Show how speakers are distributed across `age_group`, `sex`, `ethnicity`
    * Show summary stats for `duration_seconds` and `mean_f0_hz`
    * Provide interpretation hints

* `README.txt` with step-by-step instructions.

**How the student should use it:**

1. Unzip the file.
2. Open a terminal in `lesson3_trust_dataset_eda`.
3. Install pandas (if not already):

   ```bash
   pip install pandas
   ```
4. Run:

   ```bash
   python explore_trust_dataset_structure.py
   ```
5. Read the printed results carefully and answer:

   * How many speakers are there?
   * How many utterances per speaker?
   * How many neutral vs trustworthy utterances overall?
   * Do all speakers have both intents?

This gives them a **hands-on mental model** of how the real trust dataset is structured.

---

#### 11. References

* The trust dataset paper (for structure & design details):

  * *Human voices communicating trustworthy intent: A demographically diverse speech audio dataset* – Scientific Data (2025).
* Basic pandas tutorials for high school students:

  * Intro to `DataFrame`, `.head()`, `.shape`, `.describe()`, `.groupby()`, and `value_counts()`.
* Simple dataset/EDA resources:

  * “What is a dataset?” articles and videos that explain rows, columns, and metadata.

---

#### 12. Additional Information / Teacher Tips

* **Have them draw the table**
  Ask the student to draw on paper a small mock-up: 3 speakers × 4 utterances, with columns for metadata and labels. Visualizing helps a lot.

* **Connect to fairness early**
  You can already ask: “How would we notice if there were only young speakers?” or “Why is that a problem?” This sets up later discussions on bias and fairness in AI.

* **Build habits for EDA**
  Encourage them to always start a new project with:

  * `df.head()`
  * `df.shape`
  * `df.dtypes`
  * `df.describe()`
  * `df['label'].value_counts()`

