# IV. How We Built and Tested Our Models (Explained Simply)

In this section, we'll walk through the five different computer programs we built to figure out whether someone's voice sounds "trustworthy" or "neutral." Think of these programs like five different judges, each with their own way of listening. We want a judge that is both **accurate** (gets the right answer often) and **fair** (doesn't favor or disadvantage people based on their ethnicity, age, or sex).

Here's the roadmap:
- **Section IV-A**: What we're trying to do, and how we measure "fairness."
- **Section IV-B**: Two simple, older-style models (Random Forest and Logistic Regression) — our starting point.
- **Sections IV-C and IV-D**: Two more powerful AI models (ANN and 1D-CNN) that do better.
- **Section IV-E**: Our main invention, **DANN-Trust**, a special AI that is trained to *ignore* a person's ethnicity when making decisions.
- **Section IV-F**: All the ways we score and grade these models.

---

## IV-A The Problem and How We Measure Fairness

Imagine each person's voice recording is turned into a list of **60 numbers**. These numbers describe things like how high or low the voice is (pitch), how shaky or steady it sounds, and the shape of the sound waves. A tool called **VoiceLab** does this measurement for us.

Each recording also comes with two extra pieces of info:
- A **label**: either "Neutral" (0) or "Trustworthy" (1) — this is what we want to predict.
- An **ethnicity tag**: White, Black, or South Asian.

Our goal is to build a program that:
1. **Predicts trustworthy vs. neutral correctly** from the 60 numbers.
2. **Treats all three ethnic groups equally** — meaning it should be just as accurate for one group as it is for another.

### How do we measure fairness?

We use something we call the **fairness gap**. It works like this:

> Fairness gap = (best accuracy across groups) − (worst accuracy across groups)

For example, if the model is 80% accurate for White speakers, 75% for Black speakers, and 70% for South Asian speakers, the fairness gap is 80 − 70 = **10 percentage points**. A **smaller gap is better** — it means the model treats everyone more equally.

We also calculate this same gap for age groups and male/female speakers.

---

## IV-B The Two Classic Baseline Models (Random Forest and Logistic Regression)

Before showing that our new ideas work, we need to first rebuild the older approaches from a previous study (Maltezou-Papastylianou et al.) so we have a fair comparison.

### Random Forest (RF)
Imagine asking **100 different decision-makers**, each one a "decision tree" that asks yes/no questions like "Is the pitch above X?" Each tree gives its vote, and the majority wins. That's a Random Forest.
- We use 100 trees (the original paper used 126 — close enough that the result is basically the same).
- Each tree can ask up to 10 questions deep.

### Logistic Regression (LR)
This one is even simpler. It's a math formula that draws a single straight line through the data to separate "trustworthy" from "neutral." It's like trying to split a fruit basket into apples and oranges using one straight cut.

For both models:
- We **scale the numbers** so no single feature (like pitch) overpowers the others.
- We use the **same train/test split** for all our models so the comparison is fair: 80% of the data is used for learning, 20% for testing.

**The catch**: These models are simple. They can't handle complicated patterns like "high pitch *combined with* a certain voice quality, but only for certain speakers." That's why we move on to deep learning next.

---

## IV-C The Artificial Neural Network (ANN)

A neural network is like a brain made of layers of digital "neurons" that pass information forward. Each layer learns more complex patterns than the one before it.

Our ANN has **3 hidden layers**:
1. The first layer takes the 60 numbers and re-mixes them into **128 new numbers**.
2. The second layer squeezes them down to **64 numbers**.
3. The third layer shrinks them to **32 numbers** — these 32 numbers are like the network's "internal summary" of what matters for trustworthiness.
4. Finally, the network outputs a **probability between 0 and 1** for whether the voice sounds trustworthy.

We also use two tricks to keep the model from "memorizing" the training data instead of really learning:
- **Dropout**: Randomly turn off 30% of the neurons during training, like making students study without all their notes.
- **Batch Normalization**: Keep the numbers in each layer from getting wildly big or small, like keeping a chef's heat dial in a sane range.

We train it for **50 rounds** (epochs), looking at 32 examples at a time.

We picked 3 layers because we only have about 921 voice samples for training, which isn't a huge amount. Going deeper made the model worse — it started memorizing instead of learning.

---

## IV-D The 1D Convolutional Neural Network (1D-CNN)

A CNN is the type of AI usually used for images. It works by sliding a small "window" across the data, looking for patterns in nearby spots — kind of like how your eye spots a face by noticing how eyes, nose, and mouth sit next to each other.

In our case, we treat the 60 numbers as a 1D row and slide a window of **3 numbers at a time** across them.

**Honest admission**: The 60 numbers are *not* in any meaningful order. Number 5 is pitch, number 30 is something totally different, number 50 is something else again. A sliding window doesn't really make sense here — it's like trying to find a pattern in a randomly shuffled deck of cards.

So why include it? It's a **control experiment**. If a model whose superpower is "looking at nearby items" still does about as well as a fully-connected ANN, that tells us the improvement comes from **using a deeper, smarter model in general** — not from any special trick. And our results confirm this: the ANN and CNN end up within about 1 percentage point of each other.

---

## IV-E DANN-Trust: Our Main Contribution

Here's the problem with the ANN and CNN: they *can* find shortcuts. A shortcut might be:

> "If the voice has a long-term-spectrum pattern X, predict trustworthy."

But what if pattern X happens to be more common in one ethnic group than another? Then the model is secretly using ethnicity as a clue, even though we never told it to. Worse, **different random starting points** lead the model to pick *different* shortcuts. So one trained model might be fair, while the next one trained the exact same way is unfair — totally unpredictable!

### How DANN-Trust fixes this

DANN-Trust uses a clever trick called a **Domain-Adversarial Neural Network**. Imagine two teams competing inside the same brain:

- **Team Trust** (the trust head): Tries to predict whether the voice is trustworthy.
- **Team Detective** (the domain head): Tries to guess the speaker's ethnicity just from the model's internal summary.
- **The Coach** (the shared encoder): Sits in the middle and learns features that *help Team Trust* but *fool Team Detective*.

The magic ingredient is a **Gradient Reversal Layer (GRL)**. Here's how it works in plain English:
- When the Detective gets better at guessing ethnicity, normally that would teach the brain to make ethnicity even easier to detect.
- The GRL **flips the lesson backward** — so when the Detective improves, the Coach learns to make ethnicity *harder* to detect.

The result: if Team Detective can no longer figure out ethnicity from the internal summary, then no shortcut based on ethnicity can exist there. The model has been forced to be fair.

### Turning the dial slowly

We don't start the adversarial training at full strength. Instead, we **gradually turn it up** from 0 to 1 over the course of training. This way:
- Early on, the model just learns to predict trustworthiness well.
- Later on, the fairness pressure kicks in and forces it to drop any ethnicity-based shortcuts.

We train for **80 rounds** (more than the ANN/CNN's 50) because this push-and-pull training takes longer to settle down.

### Why this is a clean experiment

DANN-Trust uses **the exact same brain size** as the ANN (a 32-number internal summary). The *only* difference is the added Detective and the gradient reversal. So if DANN-Trust is more fair than the ANN, we know for sure it's because of the fairness trick — not because we made the brain bigger.

---

## IV-F How We Score the Models

For every model, we report seven kinds of scores:

1. **Overall accuracy** — what fraction of test voices we predict correctly. We also break this down by ethnicity, age group, and sex.

2. **Precision, recall, and F1-score** — fancy ways of asking "When the model says trustworthy, how often is it right?" and "Of all the truly trustworthy voices, how many did we catch?"

3. **AUC-ROC** — a score from 0.5 (random guessing) to 1.0 (perfect) that doesn't depend on where we set the "trustworthy" cutoff.

4. **Confusion matrices** — a table showing exactly which voices got classified as what.

5. **Fairness gaps** — the gap between the best-treated and worst-treated demographic group, for ethnicity, age, and sex. Lower is better.

6. **AUC fairness gap** — the same fairness check using the AUC score instead of accuracy.

7. **Probe-classifier test** (for DANN-Trust only) — a final exam: we take the model's internal 32-number summary and try to train a brand-new helper to guess ethnicity from it. If the helper does no better than random guessing (~33% for 3 groups), we've succeeded at scrubbing ethnicity out. If the helper does much better than that, we still have leakage.

### Why we train every model 5 times

We train each model **five times with different random starting points** (called seeds: 42, 43, 44, 45, 46) on the same data split. This is important because:
- If a model gets 80% accuracy with one seed and 60% with another, that's a sign it's **unreliable** even if its average looks fine.
- A model that gets the same result every time (low variation) is a **more deployable** model — you can trust it to behave the same in the real world.

A key claim of our paper is that DANN-Trust isn't just fairer on average — it's also **much more consistent** across these five runs. That's a big deal because it means engineers can actually depend on it.
