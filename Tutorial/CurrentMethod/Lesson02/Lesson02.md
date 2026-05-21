### Lesson 2 – How Voices Become Numbers: Acoustic Features 101

---

#### 1. Lesson Summary

In this lesson, the student will learn **how a human voice recording turns into numbers** that a machine learning model can understand.

We’ll focus on the kinds of **acoustic features** used in the paper, such as:

* **Pitch (F0) and pitch variability**
* **Duration**
* **Harmonics-to-Noise Ratio (HNR)**
* **Jitter & shimmer**
* **Cepstral Peak Prominence (CPP)**
* **Long-Term Average Spectrum (LTAS)**

The goal is not to memorize every formula, but to build a **clear intuition**:

> “Each audio file → a row in a table → columns are features describing how that voice sounds.”

By the end of this lesson, the student should be able to **read a feature table** and explain, in their own words, what the key features mean and how they might relate to sounding trustworthy.

---

#### 2. Key Points

* Audio is a **continuous waveform**, but computers store it as many **samples per second** (e.g., 48,000 samples per second).
* From a waveform, we can measure **pitch**, **loudness**, **duration**, and many kinds of **voice quality**.
* The paper’s dataset computes a set of **acoustic features** for each utterance (one sentence spoken).
* **Pitch (F0)** = how high or low the voice sounds; **SD F0** = how much pitch wiggles over time.
* **Jitter** and **shimmer** measure tiny random changes in pitch and loudness, related to how stable or shaky a voice is.
* **HNR** and **CPP** measure how “clear” or “harmonic” the voice is vs. noisy/rough.
* **LTAS** summarizes the average distribution of energy across frequencies (low vs high tones).
* Each utterance becomes a **feature vector**: `[mean_F0, SD_F0, HNR, shimmer, CPP, duration, ...]`.
* These vectors are used as inputs to machine learning models to predict **intent** (neutral vs trustworthy).

---

#### 3. Real-World Examples or Stories

1. **Singing a note**
   When someone sings a steady note “aaaa,” the **pitch** (F0) stays nearly constant. If they wobble, vibrato appears as **variation in F0** and maybe in loudness (shimmer).

2. **Nervous vs calm presentation**
   When you’re nervous giving a speech, your voice might shake slightly: irregular pitch (jitter) and loudness (shimmer) increase. Calm speakers often have smoother, more stable voices.

3. **Phone call in a noisy street**
   If you call someone from a crowded street, your voice may be less clear and more noisy. That’s related to **HNR** being lower (more noise mixed into the signal).

4. **“Warm” vs “thin” voice**
   Some voices feel warm and rich (more low- and mid-frequency energy), while others feel thin (more high-frequency energy). This is partly what **LTAS** and **CPP** are capturing.

5. **Time vs frequency views**
   Looking at a waveform is like looking at **how the air moves over time**; looking at a **spectrum** is like asking: “Which musical notes (frequencies) are present and how strong are they?” Both are useful to describe a voice.

---

#### 4. Terminology Explained

* **Waveform** – The raw audio signal, showing how air pressure changes with time.
* **Sample rate** – How many audio samples are taken per second (e.g., 48,000 Hz).
* **Utterance** – One spoken sentence or phrase, saved as a separate audio file.
* **Feature** – A single measurable property (number) describing the sound (e.g., mean pitch).
* **Feature vector** – A list of features for one example (row), e.g., `[F0, HNR, duration, …]`.
* **Frame** – A short slice of audio (e.g., 20–40 ms). Many features are first computed per frame and then averaged.
* **Fundamental frequency (F0)** – The main vibration rate of the vocal folds; we hear it as pitch.
* **SD F0** – Standard deviation of F0; how much pitch changes over time.
* **Jitter** – Small, random fluctuations in pitch between successive cycles.
* **Shimmer** – Small, random fluctuations in loudness between successive cycles.
* **HNR (Harmonics-to-Noise Ratio)** – Ratio of periodic (harmonic) sound to irregular noise. Higher HNR = cleaner voice.
* **CPP (Cepstral Peak Prominence)** – Another measure related to how strongly harmonic and periodic the voice is.
* **LTAS (Long-Term Average Spectrum)** – The average energy at each frequency across a whole utterance.

---

#### 5. How It Works – From Audio File to Feature Table

Imagine we have one audio file: `speaker12_sentence3_trustworthy.wav`.

**Step 1 – Load the waveform**

* Software reads the file and gets an array of numbers (samples).
* Example: `[0.01, 0.02, 0.01, -0.01, ...]` with 48,000 samples per second.

**Step 2 – Divide into frames**

* Split the signal into overlapping windows, e.g., 30 ms each.
* For each frame, we compute basic quantities:

  * Instantaneous pitch
  * Instantaneous loudness
  * Jitter, shimmer, etc.

**Step 3 – Aggregate across frames**

* For each utterance, we compute **summary statistics**:

  * Mean F0 across all frames (average pitch)
  * SD F0 (how much F0 changes)
  * Minimum and maximum F0
  * Mean HNR, mean shimmer, mean CPP
  * Duration of the utterance

**Step 4 – Save as a row in a table**

| file_name                           | mean_F0 | sd_F0 |  HNR | shimmer |  CPP | duration | intent |
| ----------------------------------- | ------: | ----: | ---: | ------: | ---: | -------: | :----: |
| speaker12_sentence3_trustworthy.wav |   195.2 |  21.5 | 11.2 |    0.32 | 15.1 |     1.73 |    1   |

**Step 5 – Repeat for all files**

* Do this for all 1,152 files, and you get a big table — the **feature matrix**.
* This matrix becomes the input to machine learning models.

In this course, we’ll mostly work with **already-extracted feature tables** (CSV files) to make things easier. But later we’ll discuss tools like **Praat**, **VoiceLab**, or Python libraries (e.g., `librosa`, `parselmouth`) that can compute features from raw audio.

---

#### 6. Practice Exercises

**Exercise 1 – Matching Concept to Number**
For each concept, choose the most relevant feature:

a) How high or low the voice sounds overall.
b) How clean vs noisy the voice is.
c) How shaky the pitch is.
d) How long the person talks.

Features: `mean_F0`, `HNR`, `jitter`, `duration`.

---

**Exercise 2 – Reading a Feature Table**

Suppose we have this small table:

| Utterance | mean_F0 (Hz) | HNR (dB) | duration (s) | intent_label |
| --------- | ------------ | -------- | ------------ | ------------ |
| A         | 170          | 9.0      | 1.8          | 0            |
| B         | 195          | 11.2     | 1.6          | 1            |

a) Which utterance has the **higher pitch**?
b) Which utterance has the **cleaner voice quality**?
c) Which one is labeled as trustworthy?

---

**Exercise 3 – Why Use Multiple Features?**

Explain in 3–4 sentences why it might be **better to use several features together (pitch, HNR, CPP, etc.) instead of only one** when trying to detect trustworthy intent.

---

**Exercise 4 – Designing a New Feature**

Imagine you want a feature called `speech_rate` that measures **how fast someone is speaking** (syllables per second).

a) Which raw measurements might you need (e.g., number of syllables, duration)?
b) How would you roughly compute `speech_rate`?

---

**Exercise 5 – Thinking About Limitations**

Is it possible for two people to have **very different voices** but produce **similar feature values** (mean F0, HNR, etc.)? In 2–3 sentences, discuss what this might mean for a machine learning model trained only on acoustic features.

---

#### 7. Solutions / Model Answers

**Exercise 1 – Answer**

a) How high or low the voice sounds → **`mean_F0`**
b) How clean vs noisy the voice is → **`HNR`**
c) How shaky the pitch is → **`jitter`**
d) How long the person talks → **`duration`**

---

**Exercise 2 – Answer**

a) Higher pitch → **Utterance B** (195 Hz vs 170 Hz)
b) Cleaner voice (higher HNR) → **Utterance B** (11.2 dB vs 9.0 dB)
c) Trustworthy utterance → **Utterance B** (intent_label = 1)

---

**Exercise 3 – Sample Answer**

Trustworthiness is a complex perception. Pitch alone might sometimes correlate with sounding friendly or confident, but not always. HNR, jitter, shimmer, CPP, and duration all capture different aspects of how the voice sounds. By using many features together, the model can notice patterns like “slightly higher pitch + stable voice + clear harmonics,” which may be more predictive than any single feature alone.

---

**Exercise 4 – Sample Answer**

a) You would need at least:

* The **number of syllables** in the utterance.
* The **duration** of the utterance in seconds.

b) You could compute:

$$\text{speech\_rate} = \frac{\text{number\_of\_syllables}}{\text{duration\_seconds}}$$

This gives the average number of syllables spoken per second.

---

**Exercise 5 – Sample Answer**

Yes, two different people might have similar mean pitch, HNR, and duration values even if their voices sound different to us. This means the model might not be able to tell them apart based only on these features. It reminds us that acoustic features are a simplified representation of a very rich signal and that models can miss important cues that humans use.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why do we split audio into frames instead of analyzing the whole file at once?
   **A:** Because many features (like pitch) change over time. Frames let us measure how features behave locally, then we summarize them (mean, SD, etc.) for the whole utterance.

2. **Q:** Are jitter and shimmer always bad?
   **A:** Not exactly. High jitter/shimmer can indicate instability or strain, but some natural variation is normal. It depends on context and how much they deviate from typical values.

3. **Q:** Why does HNR matter for trust?
   **A:** Higher HNR means a clearer, more periodic voice, which might sound healthier and more controlled. People might unconsciously associate this with confidence and reliability.

4. **Q:** What’s the difference between HNR and CPP?
   **A:** Both relate to how harmonic/periodic the voice is, but they’re computed differently. HNR looks at energy ratios, while CPP looks at peaks in the **cepstrum** (a transform of the spectrum). For this course, just remember they both measure voice clarity/periodicity.

5. **Q:** Why do we need SD F0 if we already know mean F0?
   **A:** Mean F0 says where the pitch is centered; SD F0 tells us how much it **moves around**. Two speakers could have the same average pitch but very different pitch variation.

6. **Q:** Are these features specific to English?
   **A:** No. Pitch, HNR, jitter, shimmer, CPP, etc., are properties of human voice production, not of a specific language. They can be measured in any spoken language.

7. **Q:** Can we use deep learning to automatically learn features instead?
   **A:** Yes. Deep models (like CNNs on spectrograms) can learn their own features from raw audio. But hand-crafted features are still very useful, easier to interpret, and require less data.

8. **Q:** Do we need to understand the exact math formulas of these features to use them?
   **A:** For this project, no. It’s more important to know **what they represent** conceptually and how they might relate to sounding trustworthy.

9. **Q:** Are features noisy or perfect?
   **A:** They can be noisy. Background noise, recording quality, and errors in pitch detection all affect feature values. That’s why we often use robust statistics and cross-validation.

10. **Q:** What if two features are strongly correlated?
    **A:** It means they carry similar information (e.g., HNR and CPP might both reflect voice periodicity). Models like Random Forest can handle this fine; for linear models, we may need to be more careful, but it’s not a disaster.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** What does F0 represent?
a) Loudness
b) Pitch
c) Duration
d) Noise level

**Answer:** b) Pitch.

---

**Q2.** Which feature measures tiny random fluctuations in **pitch**?
**Answer:** Jitter.

---

**Q3.** Which feature measures tiny random fluctuations in **loudness**?
**Answer:** Shimmer.

---

**Q4.** What does a higher HNR generally indicate about the voice?
a) More noise
b) Less noise and clearer harmonics
c) Longer duration
d) Higher speech rate

**Answer:** b) Less noise and clearer harmonics.

---

**Q5.** True or False: Duration is an acoustic feature.
**Answer:** True.

---

**Q6.** What is a “feature vector”?
**Answer:** A list of feature values describing one example (one utterance), e.g., `[mean_F0, HNR, duration, ...]`.

---

**Q7.** SD F0 measures:
a) The average loudness
b) The average pitch
c) How much the pitch varies over time
d) How long the speech lasts

**Answer:** c) How much the pitch varies over time.

---

**Q8.** Why might LTAS be useful?
**Answer (short):** It summarizes how energy is distributed across frequencies (low vs high), which can reflect voice timbre and quality (e.g., bright vs dark voice).

---

**Q9.** If trustworthy intent tends to have **higher mean F0** and **higher HNR** in a dataset, what might that suggest?
**Answer:** That in this dataset, voices that sound more trustworthy are modeled as slightly higher-pitched and clearer (less noisy) compared to neutral voices.

---

**Q10.** True or False: Once you have the feature table, you no longer need the audio files for machine learning.
**Answer:** False – you don’t *need* them to run the same ML model, but they are still important for checking quality, extracting new features, or running different types of audio-based models.

---

#### 10. Mini Practice Project – Exploring Acoustic Features (with .zip)

**Mini Project Title:** *Exploring Synthetic Acoustic Features for Neutral vs Trustworthy Speech*

**Goal:**
Give the student a **small, friendly dataset** of acoustic features and let them:

* Inspect the feature table.
* Compare averages for neutral vs trustworthy intent.
* See how features differ between the two groups.

I’ve created a ready-to-use project folder with:

* `synthetic_acoustic_features.csv` – a small synthetic dataset (60 rows) with:

  * `mean_f0_hz` – mean pitch (Hz)
  * `sd_f0_hz` – pitch variability
  * `hnr_db` – Harmonics-to-Noise Ratio
  * `shimmer_db` – shimmer in dB
  * `cpp_db` – Cepstral Peak Prominence in dB
  * `duration_seconds` – utterance length in seconds
  * `intent_label` – 0 = neutral, 1 = trustworthy

* `analyze_acoustic_features.py` – a **fully commented** script that:

  * Loads the dataset
  * Prints the first rows and the shape
  * Shows basic statistics (`describe()`)
  * Computes mean feature values for each intent
  * Computes differences: (trustworthy − neutral)
  * Prints a correlation matrix for the features
  * Gives a short interpretation hint

* `README.txt` – explains how to install and run, and what to look for.

**How the student should use it:**

1. Unzip the file to a folder.
2. Open a terminal in the `lesson2_acoustic_features_intro` folder.
3. Install dependencies (once):

   ```bash
   pip install pandas numpy
   ```
4. Run the script:

   ```bash
   python analyze_acoustic_features.py
   ```
5. Carefully read:

   * The mean feature values for neutral vs trustworthy.
   * The difference in means.
   * The correlation matrix.

**Mini Project Extension Ideas (Optional):**

* Ask the student to:

  * Identify **which features change the most** between neutral and trustworthy.
  * Write a short explanation of **why** those features might matter.
  * Modify one feature column slightly (e.g., add 10 Hz to `mean_f0_hz` for trustworthy) and re-run to see how group means change.

---

#### 11. References

For building intuition about acoustic features:

* Introductory articles on:

  * “Fundamental Frequency (F0) in Speech”
  * “Harmonics-to-Noise Ratio in Voice Analysis”
  * “Jitter and Shimmer in Voice Quality”
* Tutorials using:

  * **Praat** (popular tool for phonetics): lots of online guides on measuring F0, jitter, shimmer.
  * **librosa** (Python audio library): introductory tutorials on reading audio and computing features.
* Any high-school-friendly videos on:

  * “What is a spectrogram?”
  * “Basic features of speech signals.”

(These don’t have to be specific links yet; we’ll introduce more concrete tools/resources in later lessons.)

---

#### 12. Additional Information / Teacher Tips

* **Use their own speech**
  You can have the student record 2–3 short sentences and talk about how features might differ between their “normal” voice and their “trustworthy” voice, even before computing anything.

* **Link to the paper’s tables**
  Show them one of the feature tables from the paper and explain that **our synthetic dataset is a simplified version** of that. The goal is to make the real paper’s tables feel less scary.

* **Connect to math gently**
  When you mention “mean” or “standard deviation,” tie it to what they already know from statistics: average vs spread. No heavy formulas needed.

