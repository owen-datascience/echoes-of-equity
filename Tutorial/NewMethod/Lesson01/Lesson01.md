# ⭐ **Lesson 1 — Understanding the Problem, Dataset & Fairness Goal**

### *Foundation for a Winning ISEF / AI Competition Project*

---

## **1. Lesson Summary**

This lesson introduces the core scientific problem your project is trying to solve:
➡️ **Demographic bias in intent recognition models**
➡️ **Why current models fail**, especially for minority speakers
➡️ **What trustworthy intent means**
➡️ **What the dataset looks like (1,152 utterances, 96 speakers)**
➡️ **Why adversarial debiasing is needed for fairness**

By the end of Lesson 1, the student will deeply understand the domain background, the data they will use, the scientific hypothesis, and the fairness gaps that must be solved—exactly what top ISEF judges expect.

---

## **2. Key Points**

* Trustworthiness perception is strongly influenced by **voice acoustics**.
* Most research suffers from **White Western Individualist Bias** (WWIB).
* The dataset includes **1,152 utterances from 96 speakers** across
  **White, Black, and South Asian** groups.
* The baseline Random Forest model only achieved **~71% accuracy**, with large gaps across ethnic groups.
* The project’s scientific goal: **reduce demographic performance gaps** while increasing accuracy.
* Deep learning on Mel-Spectrograms allows learning **universal prosodic cues**.
* Adversarial debiasing forces the model to **ignore speaker identity**.
* This project has strong real-world value: fairer speech technology + TrustTrainer app.
* ISEF judges look for **clarity of hypothesis, fairness motivation, and innovation**.

---

## **3. Real-World Examples or Stories**

* **Hiring & interviews:**
  Voice-based AI screening systems have been shown to score some accents lower in “confidence” or “friendliness,” even when content is identical.

* **Voice assistants (Siri/Alexa):**
  Studies show that these systems struggle more with **Black and South Asian English speakers**, reinforcing inequality.

* **Courtroom / law enforcement tech:**
  Poor calibration on minority voices can amplify bias.

These help the student explain the *real significance* of the project—ISEF judges value this.

---

## **4. Terminology Explained**

| Term                              | Simple Explanation                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| **Trustworthy Intent**            | Whether a speaker sounds sincere, honest, or believable.                                    |
| **Prosody**                       | Rhythm, pitch, tone—how something is said rather than the words themselves.                 |
| **Mel-Spectrogram**               | A picture of sound showing frequencies and intensity over time.                             |
| **Bias (in ML)**                  | When a model performs worse for certain groups.                                             |
| **Demographic Parity**            | Balanced performance across demographic subgroups.                                          |
| **Adversarial Debiasing**         | Training a model to perform well while *unlearning* sensitive attributes (e.g., ethnicity). |
| **Gradient Reversal Layer (GRL)** | A trick that tells the network to get worse at predicting sensitive attributes.             |

---

## **5. How It Works — Real-World Application**

### Step-by-step workflow of this project:

1. **Load raw audio (.wav)**
2. **Convert to Mel-Spectrograms** — easier for CNNs to learn patterns
3. **Build a CNN classifier**
4. **Add adversarial head (Ethnicity/Age classifier)**
5. **Reverse gradients** so feature extractor forgets demographic cues
6. **Train jointly**

   * Main head → maximize trust classification
   * Adversarial head → minimize its accuracy
7. **Evaluate fairness**

   * Compare White vs Black vs South Asian accuracy & recall
8. **Deploy in TrustTrainer app**

This connects the science → AI → fairness → real product pipeline.

---

## **6. Practice Exercises (5)**

### **Exercise 1 — Identify Bias Sources**

List 5 ways demographic bias can enter a speech dataset.

### **Exercise 2 — Dataset Breakdown**

Compute how many utterances per demographic group if the dataset has:

* 96 speakers
* 3 groups
* 12 samples per speaker

### **Exercise 3 — Prosody Examples**

Write 3 everyday sentences and describe how tone can make them more or less trustworthy.

### **Exercise 4 — Spectrogram Exploration (No coding yet)**

Find any audio file on your computer and use an online tool to visualize its spectrogram.
Describe what you notice.

### **Exercise 5 — Research Reflection**

In 150 words, explain why “White Western Individualist Bias” is harmful in AI systems.

---

## **7. Solutions**

### **Solution 1**

Bias sources:

1. Unbalanced dataset
2. Accent variety differences
3. Microphone quality differences
4. Age distribution differences
5. Cultural differences in prosody

### **Solution 2**

96 speakers ÷ 3 groups = 32 speakers per group
32 × 12 utterances = **384 utterances per group**

### **Solution 3**

Examples include tone of excitement, slow vs fast speaking, confidence, nervous pauses.

### **Solution 4**

Students should mention:

* Bright areas = strong frequencies
* Dark areas = quiet
* Speech has repeating formant shapes

### **Solution 5**

A strong reflection should touch on fairness, societal harm, and reduced accuracy for minorities.

---

## **8. Q&A (10 Common Student Questions)**

1. **Why can't we just use pitch/jitter features like the paper?**
   → Deep learning can discover *hidden patterns* that manual features miss.

2. **Why do speech models become biased?**
   → Uneven representation + models overfitting to identity cues.

3. **What if the dataset is too small?**
   → Use augmentation: noise, pitch shift, time stretch.

4. **Why Mel-spectrograms?**
   → They compress human speech frequencies in a human-hearing-like way.

5. **Why CNNs?**
   → Their filters learn patterns in images, which spectrograms basically are.

6. **What is adversarial training?**
   → A method to force the model to forget sensitive attributes.

7. **Can this project win ISEF?**
   → Yes—fairness + innovation + real app is very competitive.

8. **Why not use LLMs?**
   → This is a prosody-based task, not text-based.

9. **What if the adversarial head performs too well?**
   → Increase gradient reversal strength so the network unlearns more.

10. **How do judges evaluate fairness?**
    → They look at subgroup metrics: recall, precision, F1.

---

## **9. Quiz (10 Questions)**

**1. What is prosody?**
A. Voice tone & rhythm ✔️
B. Text meaning
C. Speaker gender
D. Noise level

**2. What is the main fairness issue in the dataset?**
Correct: **Unequal performance across ethnic groups**

**3. How many total utterances?**
Correct: **1,152**

**4. Baseline Random Forest accuracy?**
Correct: **~71%**

**5. Which representation is used for CNN input?**
Correct: **Mel-Spectrograms**

**6. What does the adversarial head predict?**
Correct: **Demographics (Ethnicity/Age)**

**7. What does the GRL do?**
Correct: **Reverses gradients to remove demographic cues**

**8. What metric compares fairness?**
Correct: **Recall gap or accuracy gap**

**9. Why audio augmentation?**
Correct: **Increase robustness & dataset size**

**10. Why is this project novel?**
Correct: **Adversarial debiasing for trust intent recognition**

---

## **10. Mini Practice Project (with downloadable .zip)**

### **Mini Project: Explore Prosody & Spectrograms**

The student will:

1. Load a sample .wav file
2. Convert it into a Mel-Spectrogram
3. Visualize it using matplotlib
4. Save the spectrogram as an image
5. Write a short interpretation

---

## **11. References**

### **From the attached project plan**

Directly used throughout this lesson. 

### **Additional learning resources**

* Librosa tutorial: [https://librosa.org/doc/latest/tutorial.html](https://librosa.org/doc/latest/tutorial.html)
* Torchaudio tutorial: [https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html)
* Fairness in ML (Google): [https://developers.google.com/machine-learning/fairness-overview](https://developers.google.com/machine-learning/fairness-overview)
* Adversarial Debiasing (ARXIV): [https://arxiv.org/abs/1801.07593](https://arxiv.org/abs/1801.07593)
* Spectrogram basics: [https://www.youtube.com/watch?v=4SHbg-U5cKY](https://www.youtube.com/watch?v=4SHbg-U5cKY)

---

## **12. Additional Information (Expert Tips for ISEF Competitiveness)**

* Judges love **clear hypotheses** → Here: *“Adversarial debiasing will reduce demographic performance gaps.”*
* They also value **domain significance** → Bias in speech tech has real societal impact.
* Documentation is key —
  Keep a **research logbook** and record every experiment.
* Charts matter:
  Show subgroup metrics **side-by-side** for clarity.
* Real-world application (TrustTrainer app) dramatically increases the project’s impact score.
