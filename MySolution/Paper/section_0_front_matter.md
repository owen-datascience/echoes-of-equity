# Front matter — Cover page, Title, Author, Abstract, Index Terms

> **For the writer.** Replace every `[FILL IN]` placeholder with the actual values for your submission. The Abstract below uses the locked multi-seed numbers from [MySolution/Analysis/results.json](MySolution/Analysis/results.json) — do not edit them unless you re-run [compare_all_models.py](MySolution/Analysis/compare_all_models.py).

---

## Cover page (S.-T. Yau / competition-style — page 1)

> *If you are submitting to the S.-T. Yau High School Science Award, this cover page mirrors the format of the Gold-Award-Paper exemplar (page 1). Other competitions or journals may require a different cover format — adapt accordingly.*

| Field                              | Value                                                                                          |
|-----------------------------------:|------------------------------------------------------------------------------------------------|
| **Student Name**                   | `[FILL IN — e.g., Owen Chen]`                                                                  |
| **High School**                    | `[FILL IN — e.g., Lincoln High School]`                                                        |
| **Province / State**               | `[FILL IN — e.g., California]`                                                                 |
| **Country / Region**               | `[FILL IN — e.g., United States]`                                                              |
| **Advisor Name(s)**                | `[FILL IN — e.g., Dr. Jane Doe]`                                                               |
| **Advisor Institution(s)**         | `[FILL IN — e.g., Stanford University]`                                                        |
| **Paper Title**                    | Echoes of Equity: Mitigating Demographic Bias in Trustworthy-Intent Speech Classification with Adversarial Deep Learning |

---

## Title and author block (paper page 1 of the body — IEEE conference style)

**Echoes of Equity: Mitigating Demographic Bias in Trustworthy-Intent Speech Classification with Adversarial Deep Learning**

`[FILL IN — Student Name]`

`[FILL IN — High School, City, State, Country]`

E-mail: `[FILL IN]`

---

## Abstract

The human voice carries social signals — including a listener's impression of *trustworthiness* — that are increasingly inputs to AI systems for customer-service routing, fraud detection and recruitment screening, but the available datasets and classifiers are built almost entirely on white Western speakers, leaving open whether published accuracy holds for under-represented groups. The recent Trustworthy Intent in Speech (TIS) Corpus of Maltezou-Papastylianou, Scherer and Paulmann (2025) [1] is the first publicly available speech-trust dataset to span three ethnic backgrounds, but its published Random Forest and Logistic Regression baselines plateau at approximately 71% and 69% accuracy and exhibit a 5-percentage-point accuracy gap between White and South Asian speakers. We reproduce these baselines on the same corpus and introduce three deep models: a three-layer Artificial Neural Network, a 1D Convolutional Neural Network, and **DANN-Trust**, a domain-adversarial network whose gradient-reversal layer pushes the shared encoder to forget speaker ethnicity. Across five random initialisations on a joint-stratified 80/20 split, the deep models lift overall accuracy to 75–77% — clearly above the 71% baseline — and more than halve the per-ethnicity accuracy gap (Random Forest 12.1pp; CNN and DANN both 5.0pp), matching the source paper's published 5pp gap *without* any fairness-specific machinery. DANN-Trust's distinct contribution is reproducibility: its seed-to-seed standard deviation on the fairness gap is 0.89pp, compared with 2.87pp for the CNN, meaning that a DANN-Trust model retrained on the same data with a different initialisation produces a markedly more predictable fairness profile — the property that makes a model deployable in regulated settings. Code, trained checkpoints, per-seed metrics and the joint-stratification protocol are publicly released to support reproducible voice-AI fairness research.

---

## Index Terms

Trustworthy intent · Voice perception · Demographic bias · Adversarial learning · Domain-adversarial neural network · Speech classification · Fairness in machine learning · Gradient reversal layer

---

## Word count

- Title: 16 words
- Abstract: 263 words (target was ~220; the slight overshoot keeps both the reproduction and the stability finding in the single paragraph — both are needed for the contribution to land)
- Index terms: 8 terms

---

## Self-check on the Abstract

The Abstract was written to satisfy every constraint of the self-review checklist in the plan:

- [x] Cites the source paper [1] in sentence 2.
- [x] States the 71% baseline numerically.
- [x] Names all three deep models introduced.
- [x] States the deep-model accuracy uplift quantitatively (75–77%).
- [x] States the deep-model fairness lift quantitatively (12.1pp → 5.0pp).
- [x] Distinguishes DANN-Trust's contribution as reproducibility (0.89pp vs 2.87pp), *not* as peak performance.
- [x] Mentions the multi-seed protocol explicitly so a reviewer knows the numbers are mean-over-seeds, not single-run.
- [x] Mentions the joint-stratified split and public release of code/metrics — the two reproducibility hooks.
- [x] Closes with an actionable artefact (public release) rather than a vague impact claim.
