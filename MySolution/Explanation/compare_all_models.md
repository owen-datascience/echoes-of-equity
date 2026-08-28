# `compare_all_models.py` — Plain English Walkthrough

## The big idea: a fair bake-off between five different AI models

Imagine you want to figure out which pizza recipe is best. If you make Recipe A on Monday, Recipe B on Tuesday, and Recipe C on Wednesday — using different flour, different ovens, different toppings each time — you can't tell which recipe is really best. Everything changed at once.

To be fair, you need to hold everything constant except the recipe itself: same flour, same oven temperature, same amount of cheese, same tasting judges.

That's what this script does for AI models. It trains **five different classifiers** on the *exact same* speech data, with the *exact same* train/test split, and asks: which one is most accurate, most fair across ethnic groups, and — once you retrain each model many times — are those advantages *reliable* or just lucky one-run flukes?

The five contestants:

| Model | What it is | Why it's in the bake-off |
|---|---|---|
| **RF** — Random Forest | 100 decision trees voting together | Baseline from the original TIS paper |
| **LR** — Logistic Regression | A linear decision boundary | Simpler baseline from the original paper |
| **ANN** — Artificial Neural Network | 3 fully-connected hidden layers with BatchNorm + Dropout | Owen's first "new method" |
| **CNN** — 1D Convolutional Network | Two Conv1D layers sliding filters across features | Owen's second "new method" |
| **DANN** — Domain-Adversarial Network (no BN) | Encoder that *fights* against predicting ethnicity — **no batch normalization** | Owen's headline contribution |

Each model is trained **50 times** with different random starting points (seeds 42–91). That's **5 × 50 = 250 total training runs**. It sounds like a lot, but it's the number needed to run real statistical tests on the results. Earlier versions of the paper used 5 seeds and got misleading "stability" numbers; 50 seeds is enough to tell noise from signal.

**Important detail about DANN:** the encoder in this script's `run_dann` deliberately **omits BatchNormalization** (see the code comment at line 196). This design comes from the ablation study (`run_dann_ablation.py`), which found that adding BatchNorm to a DANN encoder actually *hurts* trust accuracy (p=0.002), *widens* the ethnicity gap (p=0.011), and *inflates* the gap variance (p=0.049). The no-BN DANN is what the paper reports as "DANN-Trust."

---

## Walking through the code, section by section

### Section 1 — Setup and settings (lines 19–39)

```python
SEEDS = list(range(42, 92))    # 50 seeds: 42, 43, ..., 91
SPLIT_SEED = 42                # But ONE fixed train/test split for all models
```

Notice the difference between `SEEDS` (many) and `SPLIT_SEED` (just one):

- `SPLIT_SEED = 42` decides *which speakers go in train vs test*. This is fixed forever — same split for every model, every run.
- `SEEDS` decides *how the model's weights are randomly initialized* at the start of training. This varies 50 times per model.

Why? If the split changed too, you couldn't tell whether "DANN got a different answer" was because DANN really behaves differently or because it just happened to see easier test examples that round. Fixing the split isolates the *model's* variability from the *data's* variability.

The code comment explains why 50 (not 25): with 25 seeds, the sex-gap comparison landed at p≈0.07 — tantalizingly close to significant but not quite. Doubling to 50 gives the test enough statistical power to cross p<0.05 if the effect is real.

### Section 2 — The Gradient Reversal Layer building block (lines 41–72)

This is the "magic ingredient" used later only by DANN. Full explanation lives in [`run_dann_ablation.md`](run_dann_ablation.md), but the short version:

- Forward pass: pass information through unchanged.
- Backward pass: **flip the sign** of the correction signal.

This is what causes DANN's encoder to try to *hide* ethnicity from its own output, while still predicting trustworthy intent accurately. Think tug-of-war between two students grading the same paper — one insists the paper is about trust, the other insists it's about the writer's identity, and the encoder has to please both.

The `LambdaScheduler` callback slowly turns up the tug-of-war strength from 0 (nothing) to 1 (full pressure) over 80 epochs, so the encoder can learn *something useful* before the adversary starts pushing back.

### Section 3 — Load the data ONCE, split ONCE (lines 75–127)

```python
df = pd.read_csv(CSV_PATH)
X_df = df.drop(columns=META_COLS)     # drop names, IDs, etc.
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(X_df.mean())

strata = (y_trust.astype(int) * 10 + y_ethn).astype(int)
train_test_split(..., test_size=0.2, stratify=strata, random_state=SPLIT_SEED)
```

Three important things happen exactly once, outside the model loop:

1. **Read the CSV** and turn word labels into numbers (`Neutral → 0`, `Trustworthy → 1`; `White → 0`, `Black → 1`, `South_Asian → 2`).
2. **Clean the data:** replace infinity values (from division-by-zero in voice quality math) with NaN, then fill NaNs with the column mean.
3. **Split the data 80/20** with *joint stratification* — meaning both intent AND ethnicity proportions are preserved in each half. Without this, one side of the split could accidentally end up with mostly White speakers and skew the fairness gap.

After this section, every model sees the *identical* `X_tr_s / X_te_s / yt_tr / yt_te / ye_tr / ye_te` arrays. That's the fairness guarantee.

The script also prints a per-cell breakdown of the test set:

```
Test set per (ethnicity, intent) cell:
  White         Neutral= 48   Trustworthy= 48
  Black         Neutral= 34   Trustworthy= 34
  South_Asian   Neutral= 33   Trustworthy= 34
```

This lets a reviewer confirm the split is actually balanced before believing any downstream numbers.

### Section 4 — Five model trainers, one for each contestant (lines 130–230)

Each trainer follows the same recipe:

```python
def run_XXX(seed):
    _seed_all(seed)              # 1. Fix the randomness for this run
    model = ...build a model...  # 2. Assemble the architecture
    model.fit(X_tr_s, yt_tr)     # 3. Train on the same train set
    p = model.predict(X_te_s)    # 4. Predict on the same test set
    return probs, preds          # 5. Return probabilities AND hard predictions
```

Every runner returns the same shape — probabilities and 0/1 predictions on the shared test set. That uniform contract is what makes the evaluation code downstream simple: it doesn't need to know which model produced the numbers.

Two subtle bits worth calling out:

- **`run_rf`** has a side effect: it appends RF's feature-importance vector to a module-level list `rf_fi_per_seed`. Later this becomes the mean feature importance saved into `results.json`, which the ablation script reads to build its `dann_top15` variant.
- **`run_dann`** (line 195) is the paper's headline architecture:
  - 80 epochs (vs 50 for ANN/CNN) because the adversarial tug-of-war needs more time to settle.
  - Uses the `LambdaScheduler` from Section 2 to slowly ramp up the adversarial pressure.
  - **Encoder omits BatchNormalization** — the encoder is just `Dense(128) → Dropout → Dense(32, name="encoder_output")`. The code comment cites the ablation p-values that justify this choice.

### Section 5 — The evaluation helpers (lines 233–288)

```python
def _slice_metrics(group_arr, group_values, probs, preds):
    per_group = {...}                   # per-ethnicity/age/sex numbers
    gap_pp = (max(accs) - min(accs)) * 100
    return per_group, gap_pp

def _eval_one(probs, preds):
    # Overall accuracy + per-ethnicity, per-age, per-sex breakdowns + confusion matrix
```

Given one model's `probs` and `preds`, these helpers compute:

- **Overall accuracy** — fraction of test utterances the model got right.
- **Overall AUC** — how well the model ranks Trustworthy above Neutral (threshold-free).
- **Per-ethnicity** accuracy and AUC (the paper's primary fairness axis).
- **Per-age** and **per-sex** accuracy and AUC (secondary demographic slices — the sex axis turns out to give a headline finding of its own).
- **Fairness gap `gap_pp`** — for each demographic axis, `max group accuracy − min group accuracy`, in percentage points. Smaller = fairer.
- **Confusion matrix** — a 2×2 table of true positives, false positives, etc., useful later for error-type analysis (per-model, averaged across seeds).

### Section 6 — The main loop, with resume support (lines 291–391)

Because training 250 models takes hours, the script is careful not to lose progress:

**Step A: Load prior work from disk.**

```python
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        prior = json.load(f)
    for k, v in prior.items():
        if k.startswith("_"):
            all_results[k] = v            # keep _feature_names etc.
            continue
        if isinstance(v, dict) and v.get("seeds") == SEEDS:
            all_results[k] = v            # keep, skip retraining
            if k == "RF" and "feature_importances_per_seed" in v:
                rf_fi_per_seed[:] = v["feature_importances_per_seed"]
```

The `v.get("seeds") == SEEDS` check is the safety net. If you switch from 25 seeds to 50, the cached 25-seed results have a `seeds` field of `[42, ..., 66]` that doesn't match your new `[42, ..., 91]`, so the cache is ignored and the script starts fresh. If you don't change seeds, it happily resumes.

The RF branch also restores the `rf_fi_per_seed` side-effect list, so the ablation script's top-15 feature selection still works even when RF is loaded from cache.

**Step B: Train each model that isn't already cached.**

```python
for name, fn in MODELS:
    if name in all_results:
        print(f"[skip] {name}: already in {OUT_PATH}")
        continue
    for s in SEEDS:                                # 50 seeds
        probs, preds = fn(s)
        r = _eval_one(probs, preds)
        per_seed.append(r)
```

For each new model, run it 50 times, evaluate each time, and stack the results.

**Step C: Aggregate mean ± std across seeds.**

```python
all_results[name] = {
    "n_seeds": len(SEEDS),
    "seeds": SEEDS,
    "overall_acc_mean": float(np.mean(overall_accs)),
    "overall_acc_std":  float(np.std(overall_accs)),
    "gap_pp_mean":      float(np.mean(gaps)),
    "gap_pp_std":       float(np.std(gaps)),
    "gap_age_pp_mean":  float(np.mean(gaps_age)),
    "gap_sex_pp_mean":  float(np.mean(gaps_sex)),
    ...
    "raw_per_seed":     per_seed,      # keep individual runs for stats tests later
}
```

The `raw_per_seed` list is important: it keeps every single one of the 50 runs' numbers, not just the mean. That's what makes Section 9's statistical tests possible — Mann-Whitney U and Levene's test both need the per-seed arrays, not summary statistics.

**Step D: Save to `results.json` after every model.**

```python
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"[saved] {OUT_PATH}")
```

Ctrl-C during ANN training? Rerun the script — RF and LR (which finished earlier) get skipped, ANN restarts. Delete `results.json` to force a total redo.

### Section 7 — Print the results tables (lines 394–458)

Three tables print to the terminal:

1. **Overall performance** — accuracy, AUC, and ethnicity fairness gap for each model.
2. **Per-ethnicity accuracy** — a matrix showing each model's score on White / Black / South Asian speakers.
3. **Per-age and per-sex accuracy** — same layout but for the other two demographic slices, plus each axis's max-min gap.

Every cell is `mean ± std` across the 50 seeds. These tables are what the paper's Tables 2, 3, and 4 come from. The print loops skip metadata keys (anything whose name starts with `_`) so `_feature_names` doesn't crash the pretty-printer.

### Section 8 — Feature importance + final save (lines 460–475)

```python
if rf_fi_per_seed and "feature_importances_mean" not in all_results.get("RF", {}):
    fi_stack = np.array(rf_fi_per_seed, dtype=float)
    all_results["RF"]["feature_importances_per_seed"] = rf_fi_per_seed
    all_results["RF"]["feature_importances_mean"] = fi_stack.mean(axis=0).tolist()
    all_results["RF"]["feature_importances_std"] = fi_stack.std(axis=0).tolist()
```

If RF was retrained this session, its feature importances were already saved after Step D. If RF was loaded from cache and the cache didn't have importances, this block computes and adds them. Either way, `results.json` ends up with RF's per-feature importance scores, which the ablation script reads to pick its top 15.

### Section 9 — Statistical tests (lines 477–end)

This is where the script turns "DANN looks kind of stable to me" into publication-defensible claims (or, sometimes, into "sorry, the effect isn't real"). At 50 seeds, three families of tests get printed:

**A. Ethnicity gap VARIANCE (Levene's test).** Compares CNN vs DANN, ANN vs DANN, RF vs DANN. Answers: *"Is DANN's fairness-gap variance really smaller than CNN's?"* At 5 seeds an earlier version of this paper reported yes (std 0.89 vs 2.87). At 50 seeds the effect vanishes (std 3.11 vs 2.70, p=0.33). The lesson: a stability claim needs enough seeds to survive.

**B. Sex gap MEAN and VARIANCE.** The corpus wasn't originally designed to control for sex, but a persistent ~9-10 pp Female > Male accuracy gap shows up in every model. The script runs:

- **Mann-Whitney U** on RF vs each deep model + CNN vs DANN. This is a rank-based test on means — non-parametric, so it's robust to LR's zero variance. Answers: *"Do deep models actually reduce the sex gap vs baselines?"* (Yes: RF vs CNN p=0.003, RF vs DANN p=0.002.)
- **Levene's test** on the same pairs. Answers: *"Are the deep models more or less stable on the sex axis?"* (They're significantly noisier — trade-off.)

**C. Age gap MEAN and VARIANCE.** Same shape as the sex-gap section — Mann-Whitney on means, Levene's on variances. Answers whether any deep model beats RF on the age axis. (Only ANN does, p=0.046.)

```python
try:
    from scipy import stats
except ImportError:
    print("\n[warn] scipy not installed; skipping Levene's test.")
```

`scipy` ships as a scikit-learn dependency, so it's available. If it somehow isn't, the whole statistics block is skipped gracefully rather than crashing.

Each test prints `mean_a`, `mean_b` (or `std_a`, `std_b`), the test statistic, the p-value, and a verdict (`SIGNIFICANT` or `not significant`). Those p-values are what the paper's abstract and Section IV-E discussion cite.

---

## The story this script tells the paper

By the end of the run, `results.json` contains everything needed to fill in the paper:

- **Table 2** (overall accuracy / AUC / ethnicity gap) → from the `overall_*` and `gap_pp_*` fields.
- **Table 3** (per-ethnicity accuracy) → from the `per_eth_acc_mean/std` fields.
- **Table 4** (per-ethnicity AUC) → from `per_eth_auc_mean/std`.
- **Table 5** (accuracy std, gap std) → from `overall_acc_std` and `gap_pp_std`.
- **Figure 6** (per-ethnicity accuracy bars) and **Figure 7** (fairness-accuracy Pareto) → made by `make_figures.py` reading the same JSON.
- **Figure 8** (per-age and per-sex bars) → same, from `per_age_*` and `per_sex_*` fields.
- **Abstract's statistical claims** (accuracy p=0.86, ethnicity gap p=0.26, sex gap p=0.73, sex-gap RF vs CNN p=0.003, RF vs DANN p=0.002) → all printed by Section 9.

The ablation table (Table 6 in the paper) comes from the separate `run_dann_ablation.py`, but the two scripts share the same 50-seed protocol so their numbers are directly comparable.

---

## In one sentence

**`compare_all_models.py` trains five models × fifty seeds on the exact same speech-data split, saves progress after each model so long runs are safe to interrupt, and prints Mann-Whitney U and Levene's-test p-values on the ethnicity / sex / age fairness axes — turning every headline claim in the paper (and the retraction of the old "DANN is more stable" claim) into a defensible statistical statement.**
