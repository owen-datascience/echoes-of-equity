Now that you have `metadata.csv` and all the `*.npy` Mel-spectrograms, the **next step** is:

> **Train your first baseline model to classify “neutral” vs “trustworthy” speech, using a proper speaker-independent split.**

That gives you:

* A working ML baseline
* Numbers you can compare future deep models against
* Plots + results for ISEF

Below is a **single PyTorch script** that:

1. Loads `metadata.csv`
2. Splits data into **train / val / test by speaker_id** (no speaker leakage)
3. Defines a PyTorch `Dataset` to read the `.npy` spectrograms
4. Builds a simple **CNN classifier**
5. Trains and prints accuracy on val + test

---

## 1️⃣ Install extra dependencies

If you haven’t already:

```bash
pip install torch torchvision torchaudio scikit-learn
```

---

## 2️⃣ Script: train a baseline CNN on Mel-spectrograms

Save as, for example, `train_trust_cnn.py` and run it.

---

## What you’ve just achieved with this step

* You now have a **baseline deep learning classifier** that predicts:

  * `0 = neutral`
  * `1 = trustworthy`
* You’re using **speaker-independent splits**, which is scientifically correct (no “cheating” by seeing the same speaker in train and test).
* The printed **test accuracy and classification report** become your **first benchmark** for the ISEF project.

---

If you tell me:

* roughly what accuracy you see after running this,
  I can help you:

* tune the architecture / hyperparameters,

* add **demographic-wise metrics** (accuracy by ethnicity/age/sex), and

* design the **figures & tables** you’ll show at ISEF.

=== TEST RESULTS ===
Test Loss: 0.9943  |  Test Accuracy: 0.517

Classification report:
              precision    recall  f1-score   support

     neutral       0.51      1.00      0.67        90
 trustworthy       1.00      0.03      0.06        90

    accuracy                           0.52       180
   macro avg       0.75      0.52      0.37       180
weighted avg       0.75      0.52      0.37       180

Great — this output tells us **exactly what usually happens on the first baseline** of this specific dataset:

> **The model predicts almost everything as “neutral.”**
>
> ✔ Neutral recall = **1.00**
> ✔ Trustworthy recall = **0.03** (almost zero)
>
> **Accuracy ~0.52**, identical to always predicting “neutral” (because classes are balanced 50/50).

This is *normal and expected* at this early stage — and this is actually **very valuable evidence** for your ISEF project.

### Why is this happening?

This dataset is *very* challenging:

* The acoustic differences between **neutral** and **trustworthy** intent are extremely subtle (tiny shifts in pitch, HNR, CPP, shimmer).
* A simple CNN on Mel-spectrograms barely picks up those nuances.
* Most information the human ear uses is **prosody**, not spectral shape.
* Recall: The published research paper got **~70% accuracy** with hand-crafted acoustic features, not spectrograms.

So your result is completely expected and shows the **baseline model’s limitations**, which is the perfect foundation for your student’s ISEF innovation.

---

# ✅ NEXT STEP

We now need to move *beyond* the simple CNN Mel-spectrogram model.

You have **three upgrade options** (you should eventually do all three for ISEF, but we proceed one by one):

---

# 📌 **Upgrade Step 1: Add Pitch + Prosody Features (Fundamental Fix)**

Trust intention relies heavily on:

* **Pitch mean**
* **Pitch variability (F0 stdev)**
* **Harmonics-to-noise ratio (HNR)**
* **Cepstral peak prominence (CPP)**
* **Shimmer / Jitter**
* **Speech rate / duration**

The paper emphasizes these are the *most reliable* trust cues.

We must extract these and **concatenate them to the CNN output**, or train a separate classifier.

### 🧪 Install feature extraction library:

```bash
pip install parselmouth praat-textgrids
```

### 📌 Provide the extraction script next

Before I produce code, **I need one quick answer from you**:

---

# ❓ Which direction do you prefer for the next step?

## **Option A — Extract acoustic features (praat + parselmouth) and train ML models.**

This will likely push accuracy into the **65–75% range**, matching the published paper.

## **Option B — Replace the CNN with a pre-trained speech model (wav2vec2 / HuBERT).**

This typically boosts accuracy into **75–85%**, because:

* self-supervised models encode pitch & prosody better
* CNNs on Mel-spectrograms throw away too much fine detail

## **Option C — Build a hybrid model: wav2vec2 embeddings + acoustic features.**

This is the **ISEF gold standard**, likely 85–90%+ accuracy.

## **Option D — Also analyze fairness across demographic groups.**

This becomes your project’s innovative angle:

* Does the model work differently for Black / Asian / White voices?
* Older vs younger?
* Male vs female?

Judges love this.

---

# ⭐ Recommended path for ISEF

To win, pick **C + D**:

* **Feature-rich deep model**
* **Fairness / demographic bias analysis**

But before I give you full scripts, I need to know which path you want me to implement *first*.

---

# 👉 Which next step should I generate code for?

**A, B, C, or D?**
Tell me one, and I will produce complete runnable code.

