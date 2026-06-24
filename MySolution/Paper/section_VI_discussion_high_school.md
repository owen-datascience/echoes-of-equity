# VI. Discussion: What Our Results Really Mean (Explained Simply)

Now that we've shown all the numbers in Section V, let's step back and talk about what they actually mean. This section has three parts:
- **VI-A**: The three things that worked — our wins.
- **VI-B**: The trade-off between being accurate and being fair.
- **VI-C**: The limitations of our study — what we still need to fix or improve.

---

## VI-A What Worked: Our Three Wins

### Win #1: Deeper Models Are More Accurate

Every deep learning model beat the old-school models:
- ANN was **3.1 percentage points better** than Random Forest.
- CNN was **2.0 percentage points better**.
- DANN-Trust was **1.6 percentage points better**.

And this wasn't a one-time fluke — the improvement showed up consistently across all 5 training runs.

**What this tells us**: The 71% accuracy that previous research kept hitting wasn't a true "ceiling" of what's possible. It was a ceiling for **simpler models only**. The voice features actually contain more useful patterns than simple models can pick up. Once we use a more powerful AI, those hidden patterns become accessible.

Think of it like this: a beginner chess player might plateau at a certain skill level. But that doesn't mean chess itself is "solved" at that level — a more skilled player would see deeper strategies.

### Win #2: Deeper Models Are Also Fairer (Without Even Trying!)

This was the most surprising finding of our paper. The fairness gap between ethnic groups **dropped by more than half** just by using deeper models:
- Random Forest had a **12-point** gap.
- CNN had only a **5-point** gap.
- DANN-Trust had a **5-point** gap.

And we didn't add any special fairness mechanism to the ANN or CNN! We just made them deeper and smarter, and they naturally became fairer.

The South Asian group benefited the most — their accuracy jumped from 66.6% (Random Forest) up to about 75% (CNN and DANN-Trust). That's roughly **8 percentage points of improvement** for the group that was being treated worst.

**Why is this surprising?** Most fairness research assumes you need special tricks to make a model fair. Our results suggest something simpler: **for this particular dataset, the old models were unfair partly because they weren't smart enough**. They were taking shortcuts based on demographics because they couldn't see the real patterns. Smarter models could see past those shortcuts on their own.

### Win #3: DANN-Trust Makes Fairness Predictable

CNN and DANN-Trust look almost identical on average:
- CNN: 75.4% accuracy, 4.97 pp fairness gap.
- DANN-Trust: 75.0% accuracy, 5.03 pp fairness gap.

So why bother with DANN-Trust? Because of **consistency** across retraining:
- CNN's fairness gap **wobbles by ±2.87 pp** between training runs.
- DANN-Trust's fairness gap **wobbles by only ±0.89 pp** — about one-third as much.

**What this tells us**: When you train a regular CNN, each random starting point causes it to discover **different** ethnicity-based shortcuts, leading to wildly different fairness behavior each time. DANN-Trust's special "Team Detective vs. Team Trust" training makes all these shortcuts equally unattractive, so every training run lands in roughly the same fair place.

### Bonus Finding: We Confirmed What Features Matter Most

We checked which voice features the Random Forest model considered most important, and we got the **same answer** as the original paper:
- F0 (pitch) average
- F0 variation
- HNR (voice harshness)
- Shimmer (voice shake)
- CPP (voice clarity)

Some other feature families (called LTAS) didn't matter much. This independent confirmation strengthens our confidence in these acoustic findings.

---

## VI-B The Accuracy vs. Fairness Trade-off

Figure 13 in our paper plots every model on a graph:
- **Horizontal axis**: overall accuracy (further right = more accurate).
- **Vertical axis**: fairness gap (lower = more fair).
- **The "good corner"** is the bottom-right (high accuracy AND low gap).

Here's what each model looks like:
- **Random Forest and Logistic Regression** sit in the top-left — modest accuracy, big fairness gap. The bad corner.
- **ANN** sits in the top-right — highest accuracy but still a big gap (over 10 pp).
- **CNN and DANN-Trust** sit in the bottom-right — the good corner. Both are accurate AND fair.

### So how do CNN and DANN-Trust compare?

They land in nearly the same spot on the graph. But the **error bars** tell a totally different story.

The error bars show how much each model **wobbles** between training runs:
- **CNN's vertical error bar (fairness wobble)** stretches about ±3 pp. So when you actually train a CNN, the fairness gap could be anywhere from 2 pp (great!) to 8 pp (bad). It's a coin flip.
- **DANN-Trust's vertical error bar** is about ±1 pp. The fairness gap stays in a tight 4-to-6 pp range no matter how you train it.

**The bottom line**: These two models look identical on paper, but they're really not the same product.
- CNN's "5 pp fairness gap" is an *average* that depends on lucky random seeds.
- DANN-Trust's "5 pp fairness gap" is a *promise* that holds up every time.

For real-world systems that need to retrain regularly (which is most of them!), the predictable model is way more useful — even though it gives up a tiny 0.4 percentage points of peak accuracy.

We also marked the original paper's Random Forest result on the graph (a black star at 71% accuracy, 5 pp gap). All our deep models land to the right of it — more accurate with the same or better fairness. So our paper's headline holds up against the original work.

---

## VI-C What We Couldn't Do: The Limitations

Every research paper has limitations, and being honest about them is important. We list six in Table 10 below. Three are especially important because they shape how to interpret everything else.

### Limitation #1: Some Demographic Cells Are Tiny

The full dataset has only **8 older Black speakers** and **8 older South Asian speakers**. That's a really small sample. Any accuracy measurement on those specific groups has a lot of natural uncertainty just because there aren't many people in them.

We tried to handle this by carefully splitting the data and averaging over 5 training runs, but we can't make up for genuinely small sample sizes. The proper next step is something called "bootstrap confidence intervals" — basically, a statistical technique to honestly report how uncertain those small-group estimates are.

### Limitation #2: "Trustworthy" Was Self-Rated

When the recordings were made, speakers were asked to **sound trustworthy as they understood it**. But here's the catch: just because *you* think you sound trustworthy doesn't mean a listener actually finds you trustworthy!

So technically, our model is classifying "what the speaker was *trying* to convey" rather than "what listeners actually *hear* as trustworthy." These two might not always match. To really validate the work, future studies would need to ask actual human listeners to rate the recordings.

### Limitation #3: DANN-Trust Doesn't Fully Hide Ethnicity (Yet)

Remember the probe test from Section V? We tried to predict ethnicity from DANN-Trust's internal summary:
- Random guessing would get 33% accuracy (since there are 3 groups).
- A probe achieved **54% accuracy** — better than random.

So our adversarial training **didn't completely erase ethnicity** from the model's internal brain. About 20 percentage points of ethnicity information are still leaking through.

But here's the nuance: the trust-prediction part of DANN-Trust isn't **using** that leaked information to give one group worse accuracy. So in practical terms, the model is fair — even though it isn't theoretically perfect.

To close this remaining leak, we could try:
- A bigger or smaller internal summary size.
- A more aggressive "Team Detective" with deeper layers.
- Multiple Team Detectives, each guarding against different attributes (ethnicity, age, sex).

### The Full List of Limitations

**Table 10: Six limitations and how to fix them.**

| # | What's Limited | How We'd Fix It |
|--:|----------------|------------------|
| 1 | Tiny sample sizes for older Black (N=8) and older South Asian (N=8) speakers | Use bootstrap statistics to report honest uncertainty; collect more data from older non-white speakers |
| 2 | "Trustworthy" was self-rated, not validated by real listeners | Have listeners rate every recording; train a model to align speaker intent with listener perception |
| 3 | We only tested on one dataset | Re-run everything on a second voice-trust dataset (like a Mandarin or Latino one) once available |
| 4 | We didn't extensively tune the DANN-Trust adversarial strength setting | Run a systematic search of different settings and report how sensitive results are to each |
| 5 | We only used one specific train/test split | Add 5-fold cross-validation — try different splits and average the results |
| 6 | DANN-Trust still leaks some ethnicity info (54% probe vs. 33% chance) | Try bigger internal summaries, deeper detectives, or multiple detectives for different attributes |

The first five limitations can be solved by collecting more data or using more computing power. The sixth one is the most interesting research question this paper opens up: **if we can drop the leakage all the way to chance levels, does DANN-Trust still win on consistency?** If yes, it makes a very strong case for using adversarial debiasing in all voice AI systems going forward.
