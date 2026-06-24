# VII. Conclusion (Explained Simply)

## What Was the Problem?

Computer programs that try to tell whether a voice sounds "trustworthy" have been stuck at about **71% accuracy** for years. Even worse, the most popular model (called Random Forest) gets it wrong **5% more often for South Asian speakers** than for White speakers.

That might sound like a small difference, but think about what these programs are used for:
- Routing customer service calls
- Spotting fraud
- Screening job applicants
- Even being used as evidence in courtrooms

If the model judges some people as "less trustworthy" just because of how their voice sounds, that's real harm to real people. This is what motivated our research.

---

## What Did We Discover?

We made **three main contributions**:

### 1. We Faithfully Rebuilt the Old Models

Before claiming we did better, we first had to prove we could match the original results. We did:
- Our Random Forest hit **73.4% accuracy** (the original paper got 71%).
- Our Logistic Regression hit **71.9% accuracy** (the original paper got 69%).

Both are within 2 percentage points of the originals, which means our setup is trustworthy and any improvements we report later are real.

### 2. Deep Learning Boosted Both Accuracy AND Fairness

We built two more powerful AI models:
- **ANN (a 3-layer neural network)** reached **76.5% accuracy**.
- **CNN (a convolutional neural network)** reached **75.4% accuracy**.

Both are clearly better than the old models. But the really exciting part was fairness:
- The accuracy gap between ethnic groups dropped from about **12 percentage points** down to just **5 percentage points**.
- The South Asian group got the biggest boost — their accuracy jumped from 66.6% to about 75%.

The cool surprise: **we didn't have to add any special fairness tricks.** Just using a more powerful AI model fixed most of the unfairness on its own.

### 3. Our DANN-Trust Model Is the Most Reliable

This is our headline finding. DANN-Trust is our custom-built AI that's specifically designed to ignore ethnicity when making decisions.

On average, DANN-Trust is about the same as the CNN — 75% accuracy and a 5 pp fairness gap. So what's special?

**Consistency.** When you train the same model 5 different times:
- The CNN's fairness gap **wobbles by ±2.87 percentage points**. That means one trained version might be quite fair, but the next one could be much worse.
- DANN-Trust's fairness gap **wobbles by only ±0.89 percentage points** — roughly one-third the wobble.
- Accuracy wobble: ANN swings by ±2.53%, while DANN-Trust swings by only ±0.88%.

In other words, DANN-Trust trades a tiny bit of peak accuracy for **predictable, reliable fairness** every single time.

---

## The Whole Story in One Sentence

> Deep learning models lift accuracy from 73% to 75–76% and shrink the unfairness gap between ethnic groups from 12 points down to 5 points. Among these deep models, DANN-Trust is the only one that delivers this fairness **consistently**, with about one-third the wobble of the alternatives.

---

## Why Does This Matter?

Voice-based AI is showing up everywhere — call centers, banks, hiring tools, even courtrooms. If a model systematically misjudges Black or South Asian voices, it causes real harm to real people.

You might think: "Just don't tell the AI what ethnicity each person is!" But that doesn't work. The AI is sneaky — it can figure out demographics from clues in the voice itself, even if you never tell it. So you have to actively design the model to ignore those clues.

What our paper shows is:
- **Deeper, smarter AI models** naturally remove most of the ethnic accuracy gap on their own.
- **Adversarial training** (the trick DANN-Trust uses) makes that fairness behave **the same way every time** you retrain the model.

That second property is huge. Imagine a hospital or court has to certify a model as fair every few months. If the model could randomly become unfair after retraining, the certification means nothing. With DANN-Trust, you can actually trust the audit results.

---

## What's Next? (Five Future Ideas)

There's plenty more to do. Here are five directions we'd like to explore:

1. **Ask real listeners what they think.** Right now, our "trustworthy" labels are based on how the voice was *intended* to sound. We could add ratings from real human listeners and see whether they agree.

2. **Test on more datasets.** Does DANN-Trust still reduce variance when we try it on a totally different voice dataset (like Mandarin or Latino voices)? We need to confirm this isn't just a fluke of our specific dataset.

3. **Use raw audio instead of hand-crafted features.** Right now we feed the AI 60 numbers extracted from each voice. Newer tools (like wav2vec2 or HuBERT) can work directly on the audio. We could plug those in and keep the fairness trick.

4. **Add more demographic groups to protect.** Right now we only guard against ethnicity-based bias. We could easily extend the model to also ignore accent, native language, or socioeconomic background — each one just adds another "Team Detective" to the training game.

5. **Release a free fairness audit tool.** Build a small, easy-to-use Python package that any voice-AI team can run on their own data to measure their fairness gaps. Lower the cost of doing the right thing.

---

## Open Science

We're releasing everything — the code, the trained models, the per-run statistics, and the data-splitting protocol — so anyone can reproduce our results and build on them. Fair AI research only works if it's open.
