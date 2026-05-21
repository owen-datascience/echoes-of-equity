### Lesson 1 – Big Picture: What Is This “Trustworthy Voice” Project?

---

#### 1. Lesson Summary

In this lesson, the student will get a **high-level overview** of the research paper *“Human voices communicating trustworthy intent: A demographically diverse speech audio dataset”* and the project they will eventually reproduce.

We’ll explain in plain language:

* What the paper is trying to discover:
  *Can we hear trust in someone’s voice, and can a computer detect it?*
* What the dataset looks like:
  96 speakers of different ages and ethnicities, each reading sentences in a **neutral voice** and with a **trustworthy intent**, giving 1,152 audio files. 
* How the researchers turned raw recordings into numbers (acoustic features like pitch and loudness).
* How they trained machine learning models (Random Forest & Logistic Regression) to classify “neutral” vs “trustworthy” speech, reaching about **70% accuracy** and AUC around **0.71–0.78**. 

By the end of this lesson, the student should understand **what the final goal is** and how all the pieces fit together: recordings → features → labels → ML model → results.

---

#### 2. Key Points

* The paper provides a **public speech dataset**: 1,152 utterances (short sentences) recorded by 96 speakers. 
* Each speaker says sentences **twice**: once in their **neutral** voice and once trying to sound **trustworthy**. 
* Speakers come from **different ethnicities** (white, Black, South Asian) and **age groups** (younger 18–45, older 60+), and both sexes. 
* All recordings are **standardized** (same format and sampling rate) and cut into separate files for each sentence. 
* For each audio file, the researchers extracted **acoustic features** such as:

  * Fundamental frequency (F0, perceived as pitch)
  * Amplitude (loudness)
  * Harmonics-to-noise ratio (HNR)
  * Jitter, shimmer
  * Cepstral peak prominence (CPP)
  * Long-term average spectrum (LTAS) 
* They used these features as **inputs** to two ML models: **Random Forest** and **Logistic Regression**. 
* They evaluated models using **accuracy**, **precision**, **recall**, **F1**, and **AUC**, and got around **70% accuracy** overall. 
* The student’s long-term goal:

  * Download the dataset & code
  * Understand the steps
  * Re-run and slightly extend the experiments in Python.

---

#### 3. Real-World Examples or Stories

1. **Phone call from a stranger**
   Imagine you get a call from someone saying, “Hi, I’m from your bank, I just need your card number.”
   Even before checking the facts, your *ears* judge if they sound trustworthy: do they sound calm, confident, and helpful — or rushed and nervous?

2. **Teacher vs scammer voice**
   Compare your favorite teacher saying, “Don’t worry, I’ll help you,” with a random online scammer. Their **words** might be similar, but the **voice** feels different. That emotional “feel” is partly what this dataset captures.

3. **Virtual assistants**
   Voice assistants (Siri, Alexa, Google Assistant) are designed to sound friendly and trustworthy. Their voice pitch, speed, and clarity are tuned so people feel comfortable.

4. **Customer service**
   When you call tech support, the agent’s voice can make you feel safe (“They know what they’re doing”) or annoyed. Companies care a lot about this because it impacts customer satisfaction.

5. **Robots & social AI**
   As we design social robots or AI that talk to people (elder care robots, therapy chatbots with voice), we need to understand **what makes a voice sound trustworthy** so we don’t accidentally create creepy or misleading behavior.

---

#### 4. Terminology Explained

* **Utterance** – One spoken sentence or phrase, saved as a single audio file.
* **Neutral intent** – The speaker just talks normally, without trying to sound extra anything (friendly, scary, etc.).
* **Trustworthy intent** – The speaker *tries* to sound like someone you can trust (warm, reliable, honest).
* **Acoustic feature** – A number describing some property of the sound (e.g., pitch, loudness, voice quality).
* **Fundamental frequency (F0)** – How fast the vocal folds vibrate; we hear it as **pitch** (high vs low voice).
* **Harmonics-to-noise ratio (HNR)** – How “clean” or “noisy/rough” a voice is. Higher HNR = cleaner, less noisy voice. 
* **Jitter** – Tiny random changes in pitch, reflecting instability in the vocal folds.
* **Shimmer** – Tiny random changes in loudness, reflecting instability in vocal intensity.
* **Dataset** – A structured collection of data (here: audio files + speaker info + acoustic features + labels).
* **Label** – The thing we want the model to predict (here: neutral vs trustworthy).
* **Classifier** – A machine learning model that tries to assign each example to a category (class).
* **Random Forest** – A model that uses many decision trees and averages their predictions.
* **Logistic Regression** – A simple, widely-used linear model for binary classification.
* **AUC (Area Under the ROC Curve)** – A number between 0 and 1 that shows how well the model separates the two classes (0.5 = random, 1.0 = perfect).

---

#### 5. How It Works (Real-World Pipeline)

Here’s the **full pipeline** of the paper, simplified:

1. **Collect speech recordings**

   * 96 people speak 20 sentences each.
   * Each sentence is recorded twice: neutral + trustworthy. 

2. **Standardize the audio**

   * Convert all recordings to the same format: WAV, 48 kHz sampling rate, same loudness level (67 dB). 

3. **Segment into utterances**

   * Each long recording session is cut into **single-sentence files** so every sentence is a separate audio file.

4. **Extract acoustic features**

   * Tools like **VoiceLab** compute pitch, HNR, jitter, shimmer, CPP, LTAS, etc., for each file. 
   * Result: each audio file becomes a **row in a table**, with columns like `mean_F0`, `HNR`, `duration`, etc.

5. **Set labels**

   * Each row is labeled as **neutral** or **trustworthy** intent (`0` or `1`).

6. **Train machine learning models**

   * Use acoustic features as **inputs (X)** and intent labels as **outputs (y)**.
   * Train **Random Forest** and **Logistic Regression** using a strategy called **leave-one-speaker-out cross-validation** (train on everyone except one speaker, test on the left-out speaker; repeat). 

7. **Evaluate performance**

   * Check how often the models can correctly guess whether a file is neutral vs trustworthy.
   * Accuracy ≈ 70%, AUC ≈ 0.71–0.78 overall and across subgroups. 

8. **Interpret features**

   * Use **feature importance** (Random Forest) and coefficients (Logistic Regression) to see which acoustics matter most.
   * Pitch, pitch variability (SD F0), HNR, shimmer, and CPP come out as important for distinguishing trustworthy intent. 

In the **full 10-lesson course**, we’ll go deep into each step. Lesson 1 just builds the mental map.

---

#### 6. Practice Exercises

**Exercise 1 – Why Does Voice Matter?**
In 3–5 sentences, explain **why** you think a person’s voice (not their words) can affect whether you trust them.

**Exercise 2 – Label vs Feature**
You have this small table:

| Example | Mean Pitch (Hz) | Duration (s) | Intent      |
| ------- | --------------- | ------------ | ----------- |
| A       | 180             | 1.6          | Trustworthy |
| B       | 150             | 1.8          | Neutral     |

a) Which column(s) are **features**?
b) Which column is the **label**?

**Exercise 3 – Two Intents**
Imagine you are reading the sentence: “I will help you with your homework.”

Describe how you might say it in:
a) Neutral intent.
b) Trustworthy intent.

Mention at least **two** acoustic changes (e.g., pitch, speed, loudness).

**Exercise 4 – Pipeline Ordering**
Put these steps in the correct order for this project:

1. Extract acoustic features
2. Train machine learning models
3. Record voices
4. Evaluate model performance
5. Label each utterance as neutral or trustworthy

**Exercise 5 – Reasoning About Features**
Suppose a model learns that higher pitch and higher HNR are associated with trustworthy speech in this dataset.

Explain in 2–3 sentences why this might make sense or not make sense in real life.

---

#### 7. Solutions / Model Answers

**Exercise 1 – Sample Answer**
A person’s voice carries emotion and attitude. If someone speaks calmly, clearly, and confidently, we tend to feel they know what they’re doing and care about us. If they sound rushed, shaky, or bored, we might feel uneasy, even if their words are nice. So, our brain uses voice as a shortcut for deciding if someone is safe and trustworthy.

---

**Exercise 2 – Answer**

a) **Features**:

* Mean Pitch (Hz)
* Duration (s)

b) **Label**:

* Intent

The model uses the features (numbers about the sound) to predict the label (neutral or trustworthy).

---

**Exercise 3 – Answer**

a) **Neutral intent** – Speak in your normal everyday voice, medium speed, medium volume, not putting extra emotion into it.

b) **Trustworthy intent** – Maybe slightly slower, clearer articulation, steady volume, and a warmer tone. You might raise your pitch a little at the end to sound friendly and supportive, and avoid sounding flat or bored.

---

**Exercise 4 – Answer (Correct Order)**

1. Record voices
2. Label each utterance as neutral or trustworthy
3. Extract acoustic features
4. Train machine learning models
5. Evaluate model performance

---

**Exercise 5 – Sample Answer**

Higher pitch and higher HNR might make a voice sound more energetic, bright, and clear, which people could interpret as friendliness and honesty. However, this is not universal. In some cultures or situations, a deeper voice might be seen as more trustworthy. The model is only capturing patterns specific to this dataset, not universal laws of human trust.

---

#### 8. Q&A – Common Student Questions

1. **Q:** Why do we need both neutral and trustworthy versions of each sentence?
   **A:** Using the *same words* but different intents controls for the content. Any differences the model learns are more likely due to **how** something is said (acoustics), not **what** is said.

2. **Q:** Why not just ask people if a voice is trustworthy instead of modeling it?
   **A:** Human ratings are important, but computers can process large datasets and reveal patterns we might miss. Also, a model can be used later in applications like voice assistants or social robots.

3. **Q:** Isn’t it creepy to teach a model to sound trustworthy?
   **A:** It *can* be misused, but the goal here is scientific understanding and **ethical** applications (e.g., making customer service less stressful or helping people with communication difficulties). Ethical use is a big part of AI.

4. **Q:** What’s the difference between pitch and loudness?
   **A:** Pitch is how **high or low** a voice sounds (like musical notes). Loudness is how **strong or quiet** the sound is.

5. **Q:** What does “trustworthy intent” mean exactly?
   **A:** It means the speaker is *trying* to sound like someone you can trust — based on their own idea of what a trustworthy voice sounds like.

6. **Q:** Why do we use features like HNR, jitter, and shimmer?
   **A:** These capture aspects of **voice quality** (smooth vs rough, stable vs shaky). Past research shows these are linked to how people perceive voices (e.g., healthy vs unhealthy, confident vs nervous).

7. **Q:** Why are there different ethnicities and age groups in the dataset?
   **A:** To avoid “white western individualist bias” and to make the results more generalizable across different kinds of speakers. 

8. **Q:** What is leave-one-speaker-out cross-validation?
   **A:** It’s a way to test generalization: train on all speakers except one, test on that one, then repeat for each speaker. This checks if the model works on *new people*, not just the ones it has seen.

9. **Q:** Why not use deep learning instead of Random Forest and Logistic Regression?
   **A:** Traditional models are simpler, easier to explain, and work well with small-to-medium datasets of hand-crafted features. Deep learning might help later, but this study focuses on clear, interpretable baselines.

10. **Q:** Is 70% accuracy “good”?
    **A:** It’s better than random guessing (50%), so the model is clearly picking up real patterns. But it’s not perfect, which makes sense: trust is complex and not fully determined by voice acoustics.

---

#### 9. Quiz (10 Questions + Answers)

**Q1.** What is the main goal of the paper?
a) To recognize specific words in speech
b) To detect trustworthy intent in voices
c) To translate languages
d) To separate background noise

**Answer:** b) – The study focuses on detecting trustworthy intent in speech.

---

**Q2.** How many total audio recordings are in the dataset?
a) 96
b) 576
c) 1,152
d) 20

**Answer:** c) – There are 1,152 recordings from 96 speakers. 

---

**Q3.** What are the two main speech intents in this dataset?
**Answer:** Neutral intent and Trustworthy intent.

---

**Q4.** Which of the following is **not** an acoustic feature mentioned in the paper?
a) Jitter
b) Shimmer
c) Cepstral peak prominence (CPP)
d) Translation probability

**Answer:** d) – Translation probability is unrelated; the others are acoustic features. 

---

**Q5.** Why did the researchers include speakers from different ethnicities and age groups?
**Answer:** To make the dataset more diverse and reduce bias, so the results generalize beyond a single demographic group. 

---

**Q6.** What is the label that the models try to predict?
a) The sentence ID
b) Speaker ID
c) Age
d) Intent (neutral vs trustworthy)

**Answer:** d) – The label is the speech intent.

---

**Q7.** Which models were used to classify trustworthy intent?
**Answer:** Random Forest and Logistic Regression. 

---

**Q8.** What does an AUC of 0.75 roughly mean?
**Answer:** It means that, on average, the model has a 75% chance of ranking a random trustworthy example higher than a random neutral example — better than random (0.5), but not perfect (1.0).

---

**Q9.** Why do we standardize the audio files (same format, sampling rate, and loudness)?
**Answer:** To remove technical differences between recordings so that the model focuses on **meaningful** differences in voice, not on file format or volume.

---

**Q10.** True or False: This dataset includes actors pretending to be different characters.
**Answer:** False – The speakers are **untrained**, meaning they are not professional actors. 

---

#### 10. Mini Practice Project (With Downloadable .zip)

**Title:** *Mini Voice Trust Classifier (Synthetic Data)*

Goal: Let the student **practice the core idea** of the paper on a tiny, easy-to-run dataset:

* Each row is a synthetic “utterance” with:

  * `duration_seconds`
  * `mean_pitch_hz`
  * `hnr_db`
  * `intent_label` (0 = neutral, 1 = trustworthy)
* They will train **Logistic Regression** to predict `intent_label` and see the accuracy & model coefficients.

I’ve created a `.zip` file for you that contains:

* `lesson1_mini_voice_trust_project/mini_voice_trust_dataset.csv`
* `lesson1_mini_voice_trust_project/train_logistic_regression.py`

  * Every single line has a comment explaining what it does.
* `lesson1_mini_voice_trust_project/README.txt`

  * Step-by-step instructions to run the project.

**What the student should do:**

1. Unzip the file on their computer.
2. Open a terminal in the `lesson1_mini_voice_trust_project` folder.
3. Install dependencies (once):

   ```bash
   pip install pandas scikit-learn
   ```
4. Run:

   ```bash
   python train_logistic_regression.py
   ```
5. Read the printed **accuracy** and **coefficients**.
6. Try simple modifications, for example:

   * Change the test size
   * Remove one feature and see how accuracy changes

This mirrors the paper’s idea in a **toy environment**: acoustic features → logistic regression → intent prediction.

---

#### 11. References (Starter Reading / Watching)

* Original dataset and code (OSF repository – as cited in the paper):

  * “Trustworthy Intent in Speech (TIS) Corpora Dataset” on OSF 
* The paper itself (for later lessons):

  * *Human voices communicating trustworthy intent: A demographically diverse speech audio dataset* (Scientific Data, 2025). 
* Introductory materials (for the student):

  * scikit-learn’s “Getting Started” guide (Logistic Regression section)
  * Short YouTube videos on:

    * “What is pitch in sound?”
    * “Basic introduction to machine learning classification”
  * Any high-school-friendly overview of:

    * Random Forests
    * Logistic Regression

(We’ll bring in more targeted references — Praat, VoiceLab, detailed audio ML resources — in later lessons.)

---

#### 12. Additional Information / Teacher Tips

* **Concept first, math later**
  At this stage, don’t worry about logistic regression formulas. Focus on: *we have inputs → the model learns patterns → we can predict labels and measure how good we are.*

* **Use their own voice**
  A fun warm-up is to have the student record themselves saying one sentence neutrally and then with trustworthy intent. Let them *feel* how they change pitch, speed, and loudness.

* **Connect to ethics early**
  Raise the question: “Is it okay to build an AI that knows how to sound trustworthy?” Let them think about misleading voices, scams, and also helpful uses (therapy bots, accessibility).

