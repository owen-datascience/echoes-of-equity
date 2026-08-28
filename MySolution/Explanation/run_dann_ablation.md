# `run_dann_ablation.py` — Plain English Walkthrough

## The big idea: "ablation" = "remove one part at a time"

Imagine you baked a cake and everyone loved it. You claim the secret is the vanilla extract. How do you *prove* the vanilla matters?

You bake the cake **four more times**, each time missing one ingredient:

- One without vanilla
- One without eggs
- One without butter
- One with only your top-15 favorite ingredients

If the "no vanilla" cake is the only bad one, you've proven vanilla is doing the work. But if the "no butter" cake turns out to be *better* than the original, you've discovered something even more interesting: your original recipe had a bad ingredient in it.

That's exactly what this script does — except the "cake" is the DANN-Trust neural network, and the ablation reveals something surprising: **removing BatchNormalization from the encoder makes DANN significantly better**, not worse.

---

## What the script tests

Owen's paper claims DANN-Trust is fair *because* of the Gradient Reversal Layer (GRL). A skeptical judge will ask: *"How do you know the fairness didn't come from something else — like the dropout, or the batch normalization, or just having fewer features?"*

This script answers by training **5 versions of DANN**:

| Variant | What's removed / changed | What it tests |
|---|---|---|
| `dann_full` | Nothing (reference, WITH BatchNormalization) | Baseline to compare against |
| `dann_no_grl` | The gradient reversal trick | "Is the *adversary* the key ingredient?" |
| `dann_no_domain` | The whole ethnicity-prediction branch | "Do we even need the second head at all?" |
| `dann_no_bn` | Batch normalization | "Is BN helping — or hurting?" |
| `dann_top15` | Uses only the 15 most-important features | "Does DANN need all 60 acoustic features?" |

Each variant is trained **50 times** with different random starting points (seeds 42–91), so we can see if it's *reliably* good or just lucky. Fifty seeds is enough to run actual statistical tests (see Section 8).

**Key finding: `dann_no_bn` dominates `dann_full` on all three metrics that matter.** Removing batch normalization improves trust accuracy by 1.43 pp (p=0.002), tightens the ethnicity fairness gap by 2.13 pp (p=0.011), and shrinks the seed-to-seed variance of that gap (Levene's p=0.049). This is why the main-body DANN in `compare_all_models.py` uses the no-BN configuration — Table 6 in the paper reports this ablation as justification.

---

## Walking through the code, section by section

### Section 1 — Imports and settings (lines 20–47)

```python
SEEDS = list(range(42, 92))    # Fifty different "random starts"
SPLIT_SEED = 42                # But the same train/test split every time
EPOCHS = 80                    # Train each model for 80 passes over the data
LAM_MAX = 1.0                  # Max strength of the adversarial signal
```

Think of `SEEDS` as fifty different pens you're using to solve the same puzzle. The **puzzle** (which speakers go in train vs test) is the same for everyone (`SPLIT_SEED = 42`), but the **pen** (random initialization) changes.

Why 50 seeds and not 5 or 25? At 5 seeds, an earlier version of this analysis said "the GRL cuts fairness-gap variance by 3x!" and it turned out to be a low-seed-count artefact — the effect vanished at higher seed counts. Fifty seeds gives Mann-Whitney U and Levene's test enough statistical power to detect real effects and rule out noise. If you're just checking the script runs, temporarily change `SEEDS` to `list(range(42, 47))` for a fast 5-seed smoke test.

### Section 2 — The Gradient Reversal Layer (lines 54–82)

```python
@tf.custom_gradient
def _grl_op(x, lam):
    def grad(dy):
        return -lam * dy, None
    return tf.identity(x), grad
```

This is the "magic ingredient." Normally when a neural network learns, information flows forward (input → prediction) and errors flow backward (correction signals go back to update the weights).

The GRL does something sneaky:

- **Forward:** pass information through unchanged (`tf.identity(x)`).
- **Backward:** flip the sign of the error signal (`-lam * dy`).

The effect: the ethnicity-predictor branch tries hard to guess ethnicity, but the flipped sign tells the shared encoder to become *worse* at helping it. It's like a tug-of-war on the encoder's brain — one side pulls it toward "predict trust well," the other pulls it toward "make ethnicity impossible to detect."

The `LambdaScheduler` (lines 73–82) slowly turns up the tug-of-war strength from 0 (nothing) to 1 (full pressure) over the 80 epochs, so the encoder can learn *something useful* before the adversary starts pushing back.

Note: this GRL implementation is duplicated from `compare_all_models.py` on purpose. If we imported that module, running this script would also trigger the full 5-model comparison from scratch. Copy-pasting a small block is worth the independence.

### Section 3 — Load the data once (lines 89–121)

```python
df = pd.read_csv(CSV_PATH)
trust_enc = LabelEncoder()                # Turns "Neutral"/"Trustworthy" → 0/1
ethn_enc = LabelEncoder()                 # Turns "White"/"Black"/... → 0/1/2
```

The script reads the same CSV as every other model and does the same cleanup:

- Convert word labels to numbers (computers need numbers).
- Drop the metadata columns (ID, filename, etc.) that would be cheating to look at.
- Replace weird values (`inf` and `NaN`) with sensible fill-ins.
- Standardize features so every acoustic measurement lives on the same scale.

The split (line 111) uses **joint stratification** — both intent AND ethnicity are balanced in train vs test, so no group gets accidentally shortchanged. `split_seed = 42` matches `compare_all_models.py` exactly, which is what makes ablation numbers directly comparable to main-table numbers.

### Section 4 — The reusable "building blocks" (lines 147–168)

```python
def _encoder(inp, use_bn=True):   # The shared brain
def _trust_head(enc):             # Guesses trust vs neutral
def _domain_head(enc, num):       # Guesses ethnicity
```

Instead of copy-pasting the same code five times, the script defines three little factory functions. Each variant then just **plugs them together in a different order**. This is like having LEGO bricks — the encoder, the trust head, and the domain head are separate pieces that can be swapped in and out.

Notice `use_bn=True` on the encoder — that flag is how the `dann_no_bn` variant turns off batch normalization without duplicating the whole encoder function.

### Section 5 — The five variants (lines 174–274)

Each variant is one function. They all follow the same pattern:

1. Seed the random number generator (`_seed_all(seed)`).
2. Build a model out of the LEGO pieces.
3. Compile it (specify optimizer and loss functions).
4. Train it for 80 epochs.
5. Return the predicted probabilities on the test set.

The interesting differences:

- **`run_dann_full`** (line 174): the reference. Encoder (with BN) + trust head + GRL + domain head. Also parameterized so `dann_top15` can reuse it with a narrower feature set.
- **`run_dann_no_grl`** (line 198): keeps the domain head but *removes the GRL* — so the encoder tries to help both heads instead of fighting the domain head. Pure "multi-task" learning.
- **`run_dann_no_domain`** (line 218): deletes the domain head entirely. This is just an ANN with the DANN's smaller encoder shape.
- **`run_dann_no_bn`** (line 231): everything DANN has, but batch normalization is turned off. **This is the variant that wins the ablation.**
- **`run_dann_top15`** (line 270): only the 15 most-informative acoustic features go in.

For `dann_top15`, the script reads Random Forest's "feature importance scores" from `results.json` (which `compare_all_models.py` had to produce first — that's why line 253 raises an error if you skipped that step). RF assigns each of the 60 features a score for how useful it was; the script picks the top 15.

### Section 6 — Run everything, with resume support (lines 291–338)

Because 50 seeds × 5 variants = **250 training runs** and each takes about a minute, the total wall time is a few hours. If your laptop hibernates halfway through, you don't want to start over. So the script does two things:

**1. Load already-finished variants from disk before starting:**

```python
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        prior = json.load(f)
    for k, v in prior.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and v.get("seeds") == SEEDS:
            ablation[k] = v            # keep it, skip retraining
```

The `v.get("seeds") == SEEDS` check is important — if you switch from 25 seeds to 50 seeds, the old cache is *ignored* (its `seeds` list won't match). Changing the seed count safely triggers a full rerun. The `k.startswith("_")` check preserves auxiliary keys like `_top15_feature_indices` without treating them as variants.

**2. Save after every completed variant:**

```python
for name, fn in VARIANTS:
    if name in ablation:
        print(f"[skip] {name}: already in {OUT_JSON}")
        continue
    ...train all 50 seeds...
    ablation[name] = { ... }
    ablation["_top15_feature_indices"] = [int(i) for i in top_idx]
    ablation["_top15_feature_names"] = top_names
    with open(OUT_JSON, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"[saved] {OUT_JSON}")
```

Ctrl-C mid-run? Rerun the script — anything finished is skipped, unfinished variants pick up from scratch. To force a totally fresh run, delete `ablation_results.json` first.

For each of the 250 runs, `_eval` (line 129) computes three numbers:

- **Accuracy** — what fraction of test utterances got the right label?
- **AUC** — how good is the model at ranking trustworthy above neutral?
- **`gap_pp`** — how many percentage points separate the best and worst ethnicity group?

### Section 7 — Print the results table (lines 340–356)

The script computes the **mean and standard deviation** across the 50 seeds for each variant and prints Table 6 (the ablation table) to the terminal. Everything is already saved to `ablation_results.json` (that happened after each variant in Section 6), so the paper's figures can read from it.

### Section 8 — Statistical tests (lines 358–end)

Just eyeballing "dann_no_bn's mean acc is higher than dann_full's" is not a statistical claim. To turn it into one, the script runs **three families of tests** — because the ablation now supports three separate claims, each of which needs its own p-value.

**A. Gap MEAN comparison — Mann-Whitney U (lines 387–399).**

```python
u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
```

For every pair (`dann_full` vs each other variant), tests whether the fairness gaps come from populations with different means. Non-parametric — no assumption of normality. This is the test the paper cites for "the mean ethnicity gap tightens by 2.13 pp (p=0.011)."

**B. Accuracy MEAN comparison — Mann-Whitney U (lines 401–413).**

Same test, but on overall accuracy. This is what earned the paper's "accuracy rises by 1.43 pp (p=0.002)" claim. Without this test, the ablation would only be able to say "gap looks smaller" — with it, the paper can also say "accuracy is meaningfully higher."

**C. Gap VARIANCE comparison — Levene's test (lines 415–425).**

```python
stat, p = stats.levene(_gaps(a), _gaps(b), center="median")
```

Compares the *spread* of fairness gaps between two variants. This is the test that earned "seed-to-seed variance contracts (Levene's p=0.049)."

The three tests together are what makes the ablation a "triple-significant" finding: removing BN improves DANN on **means of two metrics** (accuracy, gap) AND on **variance of one metric** (gap stability). The paper cites all three p-values in Section IV-F.

```python
try:
    from scipy import stats
except ImportError:
    print("\n[warn] scipy not installed; skipping Levene's test.")
```

`scipy` doesn't need to be added to `requirements.txt` because it comes automatically with `scikit-learn`. If for some reason it's missing, the script prints a warning and skips the entire statistics block — it won't crash the whole run.

---

## Why the mean ± std matters (and why the story changed)

The **old** narrative was: "DANN's fairness comes from the GRL." The 5-seed ablation seemed to confirm this — variants without the GRL had wider gap variance.

The **new** narrative, once you use 50 seeds and run Mann-Whitney tests on means (not just Levene's on variances), is subtler and more interesting:

- **`dann_no_bn` beats `dann_full` on both accuracy AND fairness gap** — significantly so on both. This is a surprising finding: batch normalization, usually a helpful ingredient in neural networks, actively harms DANN. The paper's interpretation is that BN's running batch statistics get destabilized by the adversarial signal continuously reshaping the encoder — on a small tabular dataset (~921 training samples), the moving averages never settle.
- The GRL still matters (that's what `dann_no_grl` tests), but the biggest lever for improving DANN turned out to be architectural, not adversarial.

This is the story Owen tells in the paper's Table 6 and Section IV-F. This script gives him the numbers *and* the p-values to back it up.

---

## In one sentence

**`run_dann_ablation.py` trains five variants of DANN fifty times each — safely resumable if interrupted — and runs three statistical tests (Mann-Whitney U on accuracy means, Mann-Whitney U on gap means, Levene's on gap variance) so the paper can prove *with three separate p-values* that removing BatchNormalization from the encoder is a triple-significant improvement over the original DANN design.**
