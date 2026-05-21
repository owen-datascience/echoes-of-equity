# Tutorial 00 — Introduction

## The big question

> Can a computer listen to a person's voice and tell whether they sound **trustworthy** or **neutral**?

This is exactly what we will teach a computer to do. Not from the *words* the person says, but only from *how* they say them — pitch, jitter, shimmer, formants — things you might not even be able to describe in everyday language.

This is called **acoustic intent classification**, and it is part of a wider research area called **affective computing** (computers that understand human emotion and intent).

---

## Why this matters

If a system that judges "trustworthiness" works well for one group of people but poorly for another, that is **algorithmic bias**, and it has real consequences:

- Job interview screening tools
- Customer service quality scoring
- Voice-based authentication
- Court testimony analysis

The project is called **Echoes of Equity** because we want to test whether these models work fairly across different ages, sexes, and ethnicities. You will train the models — then we can examine whether they are equally accurate for everyone.

---

## The dataset

Two CSV files come with the project. We will only need the first one.

### `Speech_dataset_characteristics.csv`

This file contains roughly 1,200 rows. Each row is one audio recording of a person speaking a single sentence. The columns are organized into three groups:

#### A) Identification columns (we will drop these later)

| Column | What it is |
|--------|------------|
| `Audio_Filename` | Name of the .wav file (e.g. `1893_wof_n01`) |
| `Speaker_ID` | Unique number for each speaker |
| `Sentence_Num` | Which of the practice sentences they read |

#### B) Demographic columns (we will drop these too — for now)

| Column | What it is |
|--------|------------|
| `Speaker_Ethnicity` | e.g. White, South Asian |
| `Speaker_AgeGroup` | Younger / Older |
| `Speaker_Sex` | Male / Female |

#### C) The label — what we want to predict

| Column | What it is |
|--------|------------|
| `Speaker_Intent` | `Neutral` or `Trustworthy` |

#### D) The acoustic features — what the model learns from

These come from **Praat**, a speech-analysis tool used by linguists. Don't memorize them; just get the gist.

| Feature family | What it roughly measures |
|----------------|--------------------------|
| **F0 / Pitch** (Mean, Median, StDev, Min, Max…) | How high or low the voice is, and how much it varies |
| **Jitter** (Local, RAP, ppq5, ddp…) | Tiny instability in pitch from cycle to cycle |
| **Shimmer** (local, apq3, apq11…) | Tiny instability in loudness |
| **HNR** (Harmonics-to-Noise Ratio) | How "clean" vs "noisy" the voice sounds |
| **Formants** (F1, F2, F3, F4) | Resonant frequencies that shape vowels |
| **LTAS** (Long-Term Average Spectrum) | Overall tone color of the voice |

That's roughly **50 numeric features** per row.

> **Mental model:** every row is a short voice clip described by 50 numbers, plus one label (Neutral or Trustworthy). Our job is to teach a computer to look at those 50 numbers and guess the label correctly.

---

## What we will not do

- We will **not** process raw audio. The hard work (analyzing .wav files with Praat) has already been done — we start from the CSV.
- We will **not** ship anything to production. These models are research artifacts to study fairness, not products.

---

## The four models, ranked from simplest to fanciest

```
  Logistic Regression  ───▶  Random Forest  ───▶  ANN  ───▶  1D CNN
       (1 line)              (a forest of           (a stack       (a sliding
                              decision trees)        of neurons)    window over
                                                                    the features)
```

You will build them in that order — and along the way you will pick up the vocabulary of modern AI.

---

## Project layout

After we are done, your `MySolution/` folder will look like this:

```
MySolution/
├── ExistingMethods/
│   ├── LogisticRegression.py       ← Tutorial 03
│   ├── RandomForest.py             ← Tutorial 04
│   ├── Speech_dataset_characteristics.csv
│   └── Speaker_demographics.csv
├── NewMethods/
│   ├── ANN_Trustworthy_Intent_Project/
│   │   ├── ann_trust_model.py      ← Tutorial 06
│   │   └── Speech_dataset_characteristics.csv
│   └── CNN_Trustworthy_Intent_Project/
│       ├── cnn_trust_model.py      ← Tutorial 07
│       └── Speech_dataset_characteristics.csv
└── Tutorials/
    └── (you are here!)
```

You don't need to recreate this folder structure right away — just know it exists.

---

## A note about expectations

The published research that inspired this project found that simple models achieve about **70% accuracy** with an **AUC around 0.71–0.78**. That is not amazing! It means the task is genuinely hard. We are not trying to beat 99%. We are trying to:

1. Make a model that works *better than random* (50% on a balanced two-class problem).
2. Understand *how* each kind of model arrives at its answer.
3. Have the tools to ask: *is the model equally accurate across demographic groups?*

---

## Check yourself

Before moving on, answer these out loud:

1. What are we trying to predict?
2. What kind of input does the model see? (Hint: it is not raw audio.)
3. About how accurate should we expect the models to be?

If you can answer those, you are ready for **[Tutorial 01 — Machine Learning Basics](01_machine_learning_basics.md)**.
