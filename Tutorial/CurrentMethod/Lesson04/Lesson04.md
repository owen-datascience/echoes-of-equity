### Lesson 4 – Exploratory Data Analysis (EDA) of Trustworthy Voices

---

#### 1. Lesson Summary

In this lesson, the student will learn how to **explore the trust dataset with data science tools** before training any model.

We focus on **Exploratory Data Analysis (EDA)**:

* Checking **class balance** (how many neutral vs trustworthy utterances).
* Looking at **basic statistics** of key features (mean, std, min, max).
* Comparing **feature distributions** between neutral and trustworthy speech.
* Creating **simple histograms** to visualize differences (e.g., pitch & HNR).

The goal: build intuition like

> “Trustworthy utterances tend to have slightly higher pitch and HNR in this dataset,”

and to form good habits: **always inspect your data first**.

---

#### 2. Key Points

* EDA is the process of **getting to know your data** before modeling.
* Always check:

  * **Shape** of the dataset (rows, columns).
  * **Column names** and data types.
  * **Missing values** and obvious errors.
  * **Class balance** for the target label.
* Use **summary statistics** (`describe()` in pandas) to see:

  * Min, max, mean, standard deviation of features.
* Compare features across **groups** (here: neutral vs trustworthy; optionally across age/sex/ethnicity).
* Visual tools like **histograms and boxplots** help see:

  * Are trustworthy utterances shifted toward higher pitch?
  * Is HNR generally higher for trustworthy speech?
* EDA often reveals:

  * Outliers
  * Data entry problems
  * Potential biases
* Good EDA makes your later machine learning results **more trustworthy and interpretable**.

---

#### 3. Real-World Examples or Stories

1. **Doctor checking your vitals**
   Before suggesting any treatment, a doctor checks your **basic stats**: temperature, pulse, blood pressure. EDA is like checking the dataset’s vitals.

2. **Coach checking team stats**
   A basketball coach looks at points, rebounds, and shooting percentages for each player. They might see that one player always scores more in home games. Similarly, we might see that trustworthy utterances have higher average pitch.

3. **Quality control in a factory**
   Each product is measured; EDA is like checking the distribution of product sizes. If some are too big or too small, you know something is wrong. In our data, if durations are negative… we know something is wrong too!

4. **First impression of a new class**
   On the first day, you get a sense of how many students there are, how many like math, etc. EDA is your “first impression” of a dataset.

---

#### 4. Terminology Explained

* **EDA (Exploratory Data Analysis)** – The process of exploring a dataset to understand its main characteristics, usually using summary statistics and plots.
* **Summary statistics** – Numbers that summarize a feature: mean, median, min, max, standard deviation.
* **Distribution** – Describes how a feature’s values are spread out (e.g., centered at 180 Hz, most values between 150–210 Hz).
* **Histogram** – A bar chart that shows how many values fall into different ranges (bins).
* **Boxplot** – A plot showing the median, quartiles, and potential outliers of a feature.
* **Outlier** – A data point that is very different from most others (e.g., mean pitch of 500 Hz when others are around 150–250 Hz).
* **Class balance** – How evenly the dataset is split between labels (here, neutral vs trustworthy).
* **Groupby** – A pandas operation to compute statistics separately for different groups (e.g., by label or by age group).

---

#### 5. How It Works – EDA Step-by-Step

Suppose you load the feature table for the trust dataset (or a synthetic version).

**Step 1 – Look at the basic info**

```python
import pandas as pd

data = pd.read_csv("trust_features.csv")
print(data.head())
print(data.shape)
print(data.dtypes)
```

* `head()` shows the first few rows.
* `shape` shows how many rows and columns.
* `dtypes` shows whether columns are numbers, strings, etc.

**Step 2 – Check label distribution**

```python
print(data["intent_label"].value_counts())
```

* Are neutral and trustworthy counts similar?
* If one class is much bigger, we note this for later.

**Step 3 – Summary statistics for numeric features**

```python
numeric_cols = ["duration_seconds", "mean_f0_hz", "sd_f0_hz", "hnr_db"]
print(data[numeric_cols].describe())
```

* We see min, max, mean, standard deviation, quartiles.
* Check if any values look impossible (e.g., negative duration).

**Step 4 – Compare neutral vs trustworthy averages**

```python
print(data.groupby("intent_label")[numeric_cols].mean())
```

* We can see if, on average, trustworthy utterances have:

  * Higher `mean_f0_hz`
  * Higher `hnr_db`
  * Different `duration_seconds`

This mimics the analysis in the paper where certain acoustic features are higher/lower for trustworthy intent.

**Step 5 – Visualize distributions**

```python
import matplotlib.pyplot as plt

neutral = data[data["intent_label"] == 0]
trustworthy = data[data["intent_label"] == 1]

plt.figure()
plt.hist(neutral["mean_f0_hz"], bins=20, alpha=0.7, label="Neutral")
plt.hist(trustworthy["mean_f0_hz"], bins=20, alpha=0.7, label="Trustworthy")
plt.title("Histogram of mean_f0_hz by intent")
plt.xlabel("mean_f0_hz (Hz)")
plt.ylabel("Count")
plt.legend()
plt.show()
```

* If the “trustworthy” histogram peaks at slightly higher values, we see that trustworthy speech tends to be higher-pitched in this dataset.

**Step 6 – Reflect & hypothesize**

* Based on EDA, we form hypotheses:

  * “I expect a classifier that uses mean pitch & HNR might perform better than random.”
  * “I should be careful about speaker imbalance or potential outliers.”

These insights guide how we design and interpret our models in later lessons.

---

#### 6. Practice Exercises

**Exercise 1 – Reading Class Distribution**

You run:

```python
data["intent_label"].value_counts()
```

and get:

```text
0    180
1    180
Name: intent_label, dtype: int64
```

a) How many total utterances are in the dataset?
b) Is the dataset balanced? Explain.

---

**Exercise 2 – Interpreting `describe()`**

For `mean_f0_hz`, `data["mean_f0_hz"].describe()` prints:

```text
count    360.000000
mean     182.500000
std       22.000000
min      120.000000
25%      166.000000
50%      180.000000
75%      198.000000
max      240.000000
```

a) What is the average (mean) pitch?
b) What pitch value is at the 50% (median) mark?
c) Are there any obviously impossible values?

---

**Exercise 3 – Group Means**

You compute:

```python
data.groupby("intent_label")[["mean_f0_hz", "hnr_db"]].mean()
```

and get:

| intent_label | mean_f0_hz | hnr_db |
| ------------ | ---------- | ------ |
| 0            | 175.0      | 9.7    |
| 1            | 190.0      | 11.0   |

Describe in words how neutral and trustworthy utterances differ based on these averages.

---

**Exercise 4 – Histogram Interpretation**

You see a histogram of `mean_f0_hz` where:

* The neutral distribution mostly peaks around 170–180 Hz.
* The trustworthy distribution peaks around 185–195 Hz and is slightly shifted to the right.

What does this tell you about pitch differences between the two intents?

---

**Exercise 5 – EDA & Outliers**

Suppose in the summary stats you notice one utterance with `duration_seconds = 15.0`, while most are around 1.5–2.0 seconds.

a) Why might this be an outlier?
b) What steps might you take to handle it?

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

a) Total utterances = 180 + 180 = **360**.
b) Yes, it’s balanced; there are equal numbers of neutral (0) and trustworthy (1) utterances.

---

**Exercise 2 – Answer**

a) Average (mean) pitch = **182.5 Hz**.
b) The median (50%) pitch = **180 Hz**.
c) Values range from 120 Hz to 240 Hz, which are realistic for typical adult speech, so there are no obviously impossible values.

---

**Exercise 3 – Answer**

Neutral utterances (label 0) have an average pitch of 175 Hz and HNR of 9.7 dB. Trustworthy utterances (label 1) have a higher average pitch (190 Hz) and higher HNR (11.0 dB). So, in this dataset, trustworthy speech tends to be **higher-pitched and clearer** (less noisy) than neutral speech.

---

**Exercise 4 – Answer**

The trustworthy distribution being shifted to the right means that, overall, trustworthy utterances tend to have **higher pitch values** than neutral ones. The overlap shows they are not completely separate, but there is a noticeable trend toward higher pitch for trustworthy intent.

---

**Exercise 5 – Sample Answer**

a) It’s an outlier because almost all utterances are around 1.5–2.0 seconds, and 15 seconds is much longer than the rest. It might be a recording or segmentation error.
b) You could:

* Double-check the original audio to see if it’s a valid sample.
* If it’s clearly incorrect, remove it from the dataset.
* Or cap extremely long durations if they result from rare glitches.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why do we do EDA before training models?
   **A:** To understand the data, catch errors, see patterns, and avoid building models on bad or misunderstood data.

2. **Q:** What if the class distribution is very imbalanced?
   **A:** We might need techniques like resampling, different metrics (e.g., F1, AUC), or adjusting decision thresholds. But first, we must **notice** the imbalance through EDA.

3. **Q:** Is looking at `describe()` enough?
   **A:** It’s a good start, but visualizing distributions (histograms, boxplots) and checking relationships between features is also important.

4. **Q:** Why use histograms instead of just means?
   **A:** Means can hide details (like multi-modal distributions). Histograms show how the values are spread out, not just the average.

5. **Q:** What is a good number of bins for a histogram?
   **A:** There’s no single perfect answer; 10–30 bins is common for medium-sized datasets. The key is to see a clear shape without too much noise.

6. **Q:** Can we use boxplots instead of histograms?
   **A:** Yes! Boxplots are great for comparing medians and spread across groups and spotting outliers quickly.

7. **Q:** Should we remove all outliers?
   **A:** Not automatically. Some outliers are real and meaningful. We should investigate them and decide based on domain knowledge.

8. **Q:** Do we need to normalize features during EDA?
   **A:** For basic EDA, we usually look at raw values. Normalization is more important when training certain models (e.g., logistic regression, neural networks).

9. **Q:** How do we check relationships between two features?
   **A:** We can use scatter plots and correlation matrices to see how they move together.

10. **Q:** Does EDA change the dataset?
    **A:** EDA itself is just viewing/analyzing. But based on EDA, we might then clean or transform the dataset (e.g., removing invalid rows), which *does* change it.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** EDA stands for:
a) Exploratory Data Analysis
b) Experimental Data Analytics
c) Extended Data Aggregation
d) Exploratory Distribution Assessment

**Answer:** a) Exploratory Data Analysis.

---

**Q2.** Which of the following is *not* a typical EDA step?
a) Checking class balance
b) Computing summary statistics
c) Training a deep neural network
d) Plotting histograms

**Answer:** c) Training a deep neural network.

---

**Q3.** `data.shape` returns `(360, 12)`. What does 360 represent?
**Answer:** The number of rows (utterances) in the dataset.

---

**Q4.** A histogram is best used to:
a) Show the exact median value
b) Show the distribution of a single numeric feature
c) Show relationships between many features
d) Show text data

**Answer:** b) Show the distribution of a single numeric feature.

---

**Q5.** If `data["intent_label"].value_counts()` returns `0: 300, 1: 60`, is the dataset balanced?
**Answer:** No, it is strongly imbalanced; neutral examples are much more frequent than trustworthy ones.

---

**Q6.** What does `data[numeric_cols].describe()` *not* show?
a) Mean
b) Standard deviation
c) Median
d) Exact histogram

**Answer:** d) Exact histogram.

---

**Q7.** If trustworthy utterances have higher mean `hnr_db` than neutral ones, what does that suggest?
**Answer:** Trustworthy utterances might have **clearer, less noisy** voices in this dataset.

---

**Q8.** True or False: Seeing an impossible value (like negative duration) means we should ignore it and move on.

**Answer:** False – we should investigate and likely fix or remove it.

---

**Q9.** Which pandas method is most useful for computing means per class?
a) `.head()`
b) `.groupby()`
c) `.merge()`
d) `.drop()`

**Answer:** b) `.groupby()`.

---

**Q10.** Why might we create separate histograms for neutral and trustworthy utterances?
**Answer:** To visually compare how feature distributions differ between the two classes.

---

#### 10. Mini Practice Project – EDA & Simple Plots (Downloadable .zip)

**Mini Project Title:** *EDA and Histograms for Neutral vs Trustworthy Speech*

I’ve prepared a mini project folder that includes:

* `synthetic_trust_eda_dataset.csv`

  * Each row = one utterance, with:

    * `speaker_id`
    * `age_group`
    * `sex`
    * `ethnicity`
    * `sentence_id`
    * `intent_label` (0 = neutral, 1 = trustworthy)
    * `duration_seconds`
    * `mean_f0_hz`
    * `sd_f0_hz`
    * `hnr_db`

* `trust_eda_plots.py`

  * Fully commented script that:

    * Loads the dataset
    * Shows head, shape, columns
    * Prints class distribution
    * Prints summary stats for numeric features
    * Computes mean feature values by intent
    * Plots **two histograms**:

      * `mean_f0_hz` for neutral vs trustworthy
      * `hnr_db` for neutral vs trustworthy
    * Prints interpretation hints

* `README.txt` with run instructions.

**How the student should use it:**

1. Unzip the file.
2. Open a terminal / command prompt in `lesson4_trust_eda_plots`.
3. Install dependencies (if needed):

   ```bash
   pip install pandas matplotlib
   ```
4. Run:

   ```bash
   python trust_eda_plots.py
   ```
5. Two histogram windows will appear:

   * Compare the neutral vs trustworthy distributions for `mean_f0_hz` and `hnr_db`.
6. Answer questions like:

   * Are trustworthy utterances shifted toward higher pitch?
   * Are HNR values typically higher for trustworthy speech?

This mini project directly practices **EDA + simple plotting**, exactly what you’ll do on the real dataset later.

---

#### 11. References

* Any introductory EDA resources with pandas:

  * “Pandas `.head()`, `.describe()`, `.groupby()`, and `value_counts()`” tutorials.
* Matplotlib beginner guides:

  * “Creating histograms with matplotlib” tutorial.
* Optional deeper reading:

  * Articles or videos on:

    * “Understanding data distributions”
    * “Introduction to outliers and boxplots”

---

#### 12. Additional Information / Teacher Tips

* **Encourage curiosity**
  Let the student ask their own questions: “Do older speakers show the same pattern?” “What if we separate by sex or ethnicity?” Even if you don’t answer all of them now, this builds research thinking.

* **Teach reproducibility habits**
  Suggest that the student always keeps their EDA code in a separate script or notebook so they can re-run it when the dataset changes.

* **Link to later lessons**
  Explain that this EDA will make it easier to:

  * Understand feature importance later.
  * Interpret why the Random Forest or Logistic Regression model performs the way it does.
