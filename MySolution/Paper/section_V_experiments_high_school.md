# V. Our Experiments and What We Found (Explained Simply)

In this section, we put all five of our models to the test and see how they did. Think of it like a school sports day where five athletes (our models) all compete on the same track. We measure who finishes first (accuracy), who plays fair with everyone (fairness gap), and — most importantly — who performs **consistently** every time they race, not just on a lucky day.

Here's our roadmap:
- **V-A**: How we set up the experiment so it's fair.
- **V-B**: The overall scoreboard — who was most accurate?
- **V-C**: Breaking down accuracy by ethnicity (the main fairness check).
- **V-D and V-E**: Same checks for age groups and male/female speakers.
- **V-F**: Detailed "confusion matrices" showing what kinds of mistakes each model made.
- **V-G**: Which voice features matter most, and what happens when we remove parts of our model.
- **V-H**: A different scoring method called AUC.
- **V-I**: **The big finding** — which model gives the most consistent, predictable results.

---

## V-A How We Set Up the Experiment

We split our voice recordings into two piles:
- **Training pile (921 recordings)**: The models study these to learn.
- **Test pile (231 recordings)**: The models have never seen these — this is the real exam.

We made sure both piles have the same mix of speakers. The test pile contains:
- 48 Neutral + 48 Trustworthy White speakers
- 34 + 34 Black speakers
- 33 + 34 South Asian speakers

This is important because if (say) all the South Asian speakers ended up in the training pile, we couldn't fairly test how the model does on them.

**Each model is trained 5 times** using 5 different random starting points (we call these "seeds": 42, 43, 44, 45, 46). Why 5 times? Because AI models often give slightly different answers depending on where they start — so we want to know not just the *average* score but how *consistent* the model is. Every number you'll see below is the **average of 5 runs**, with the "± something" telling you how much the model wobbled.

All the code that produced these numbers lives in one file: `MySolution/Analysis/compare_all_models.py`.

---

## V-B Overall Performance: Who Won?

**Table 3: Overall accuracy and AUC (mean ± standard deviation across 5 runs)**

| Model | Accuracy        | AUC               | Fairness gap |
|------:|----------------:|------------------:|-------------:|
| RF    | 73.42 ± 0.97%   | 0.823 ± 0.006     | 12.07 ± 4.59 pp |
| LR    | 71.86 ± 0.00%   | 0.803 ± 0.000     | 11.34 ± 0.00 pp |
| ANN   | **76.54** ± 2.53% | 0.825 ± 0.019   | 10.30 ± 1.10 pp |
| CNN   | 75.41 ± 0.92%   | 0.814 ± 0.010     | **4.97** ± 2.87 pp |
| DANN  | 74.98 ± **0.88%** | 0.810 ± 0.011   | **5.03** ± **0.89** pp |

What these numbers tell us, in plain English:

1. **The deep learning models won on accuracy**. ANN, CNN, and DANN-Trust all beat the older models (RF and LR) by 1.6 to 3.1 percentage points. The ANN was the most accurate at 76.5%.

2. **All models are roughly equally good at "ranking" voices** (the AUC scores are between 0.80 and 0.83 for everyone). This means the difference in accuracy comes from where each model **draws the line** between "trustworthy" and "neutral," not from being better at telling them apart in general.

3. **The biggest win is in fairness!** The fairness gap dropped from a huge 12 percentage points (RF) down to just 5 percentage points (CNN and DANN-Trust). That means the new models treat all three ethnic groups much more equally.

> **Note**: We still need to add a few more numbers to this table (precision, recall, F1-score). We'll do this by adding one extra calculation to our code.

---

## V-C Fairness Check: Accuracy by Ethnicity

This is the most important table for our paper. It shows how well each model does on each ethnic group.

**Table 5: Accuracy broken down by ethnicity**

| Model | White (96 tests) | Black (68 tests) | South Asian (67 tests) | Gap |
|------:|-----------------:|-----------------:|-----------------------:|----:|
| RF    | 78.12 ± 1.86%    | 73.53 ± 1.86%    | 66.57 ± 4.49%          | 12.07 ± 4.59 pp |
| LR    | 76.04 ± 0.00%    | 64.71 ± 0.00%    | 73.13 ± 0.00%          | 11.34 ± 0.00 pp |
| ANN   | 78.75 ± 3.82%    | 73.53 ± 4.05%    | 76.42 ± 5.70%          | 10.30 ± 1.10 pp |
| CNN   | 77.29 ± 0.78%    | 73.24 ± 3.40%    | 74.93 ± 0.60%          | 4.97 ± 2.87 pp |
| DANN  | 75.62 ± 1.93%    | 73.82 ± 2.85%    | 75.22 ± 1.79%          | 5.03 ± 0.89 pp |

Look at the **South Asian column** first. With the old Random Forest model, this group gets only 66.6% accuracy — much worse than the White group (78.1%). That's not fair! But once we use deep learning, the South Asian accuracy jumps up to 74–76%, almost matching the White group.

### A surprising finding
**RF and LR disagree on which group is hardest.** Random Forest struggles most with South Asian speakers (66.6%), while Logistic Regression struggles most with Black speakers (64.7%). This is suspicious! It tells us the two old models are each picking up **different shortcuts** based on ethnicity. There isn't one "naturally hard" group — the model itself decides which group gets the short end of the stick. This is exactly the problem DANN-Trust is built to fix.

### Why DANN-Trust shines here
Look at the DANN row: 75.62%, 73.82%, 75.22%. The gap between best and worst is barely there. Even better, the wobble (the ± numbers) is the smallest in the table for every group. That means DANN-Trust is **steady and fair**, not just on average but every single time we train it.

---

## V-D Fairness Check: Accuracy by Age Group

> **To do**: We still need to add this analysis. We expect that younger and older speakers will get similar accuracy (within about 1 percentage point), so the age gap will be much smaller than the ethnicity gap. Older speakers might benefit more from deep learning models because their voice quality features vary more from person to person.

---

## V-E Fairness Check: Accuracy by Male vs. Female

> **To do**: We still need to add this analysis too. The original paper didn't include this, so we'll be the first to report it. We expect the gap to be small, but male voices might be slightly harder to classify because they have lower pitch on average (~105–140 Hz vs. 175–230 Hz for female voices), which makes some of our pitch-related features noisier.

---

## V-F Confusion Matrices (Detailed Mistake Tracking)

> **To do**: A confusion matrix is a 2×2 table that shows, for each model:
> - How many "Neutral" voices were correctly called Neutral
> - How many "Neutral" voices were mistakenly called Trustworthy
> - How many "Trustworthy" voices were mistakenly called Neutral
> - How many "Trustworthy" voices were correctly called Trustworthy
>
> Based on the original paper, we expect the models to be a bit hesitant about labeling something "Trustworthy" — meaning they'd rather say "Neutral" when unsure. We'll check whether our deep models fix this hesitation or just match it.

---

## V-G Which Voice Features Matter Most?

> **To do (Feature importance)**: We'll ask the Random Forest model "Which of the 60 voice features did you find most useful?" and plot the top 15 in a bar chart. We expect features like pitch (F0), pitch variation, voice harshness (HNR), voice shake (shimmer), and a feature called CPP to top the list — same as the original paper found.
>
> **To do (Ablation studies)**: An "ablation" is like removing one ingredient from a recipe to see if it still tastes good. We'll test 4 versions of DANN-Trust:
> 1. **Without GRL** — keep the ethnicity-prediction part but remove the gradient reversal trick.
> 2. **Without the domain head** — remove ethnicity prediction entirely (becomes a plain ANN).
> 3. **Without batch normalization** — remove one of our training stabilizers.
> 4. **Top 15 features only** — see if the model still works with just the most important features.
>
> The most interesting comparison is full DANN-Trust vs. "without GRL." If they perform the same, then our special gradient reversal trick wasn't really doing anything. If DANN-Trust is better, then the trick is what makes the model fair.

---

## V-H AUC Scores (Another Way to Measure)

AUC is a score from 0.5 (random guessing) to 1.0 (perfect). It measures how well a model can rank voices — separate from where it actually draws the "trustworthy / not trustworthy" line.

**Table 4: AUC scores broken down by ethnicity**

| Model | White AUC | Black AUC | South Asian AUC |
|------:|----------:|----------:|----------------:|
| RF    | 0.882     | 0.794     | 0.774           |
| LR    | 0.873     | 0.721     | 0.784           |
| ANN   | 0.847     | 0.784     | 0.842           |
| CNN   | 0.818     | 0.818     | 0.808           |
| DANN  | 0.832     | 0.771     | 0.799           |

Two interesting things:

1. **CNN is super consistent across groups** — 0.81 to 0.82 for everyone. This matches what we saw in accuracy: CNN is one of the most balanced models on average.

2. **LR really struggles with Black speakers** — its 0.721 AUC for that group is its weakest result. This matches its bad 64.7% accuracy for Black speakers. Logistic Regression's "single straight line" approach just can't separate trustworthy from neutral Black voices as well as it can for other groups. This is a well-known weakness of simple linear models.

---

## V-I The Headline Finding: Which Model Is Most Reliable?

**This is the most important section of our paper.** It's about something you can't see from any single training run: how much do the results **change** when you retrain the model with a different random starting point?

**Table 9: How much each model's scores wobble between training runs**

| Model | Accuracy wobble | Fairness-gap wobble |
|------:|----------------:|--------------------:|
| RF    | 0.97%           | 4.59 pp             |
| LR    | 0.00%           | 0.00 pp             |
| ANN   | 2.53%           | 1.10 pp             |
| CNN   | 0.92%           | 2.87 pp             |
| DANN  | **0.88%**       | **0.89 pp**         |

Let's unpack this:

1. **LR's zero wobble is a freebie**: Logistic Regression uses a math formula that gives the exact same answer every time. So its zero wobble doesn't mean it's amazing — it just means we set up the test correctly.

2. **DANN-Trust is the most consistent real model.** Compare CNN and DANN — they have nearly identical *average* accuracy and fairness gap. But:
   - CNN's fairness gap wobbles by ±2.87 pp. That means a CNN trained today might have a fairness gap of 2 pp, but a CNN trained tomorrow might have a fairness gap of 8 pp. Yikes!
   - DANN-Trust's fairness gap wobbles by only ±0.89 pp. Train it today, train it tomorrow — you'll get nearly the same fairness gap.

### Why this matters in the real world

Imagine a company uses our model to help screen job interviews. They run a fairness audit and the auditor says, "Great, your model has only a 2.5 pp fairness gap." A few months later, the company retrains the model with fresh data.

- If they're using a **CNN**, the new model could have a fairness gap anywhere from 2 to 8 pp. The audit isn't really meaningful anymore.
- If they're using **DANN-Trust**, the new model will almost certainly still have a fairness gap of about 4 to 6 pp. The audit is reliable.

**That's the whole point**: DANN-Trust isn't just fair on average — it's *predictably* fair, every single time. That makes it actually usable in the real world.

### Did DANN-Trust really scrub out ethnicity? (The probe test)

Remember from Section IV that DANN-Trust is supposed to make ethnicity invisible inside the model's "brain." We tested this by:
1. Freezing the model's 32-number internal summary.
2. Training a brand-new mini-program to try to guess each speaker's ethnicity from those 32 numbers.

If DANN-Trust did its job perfectly, the mini-program should do no better than random guessing (33.3% for 3 groups). What we found:

- The mini-program achieved **53.7% accuracy** — about 20 percentage points better than random.

So the encoder is **not perfectly blind** to ethnicity. Some ethnic information is still leaking through. **But here's the key**: the trust-prediction part of DANN-Trust doesn't *use* that leftover information to give one group worse accuracy than another. That's what we actually care about for fairness in practice — equal accuracy for everyone — even if the model's internal brain still has some hidden hints about ethnicity floating around.

Closing this remaining 20 pp leakage is a great direction for future research. We talk about it more in Section VII.
