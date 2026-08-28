# Step-by-Step Fix Recipe for Items 1 and 2

Concrete, paste-ready instructions for the priority-1 (paper-craft) and priority-2 (methodology) fixes from `judge_review.md`.

---

# ITEM 1 — Paper-craft fixes (an afternoon of work)

## Step 1.1 — Fix front matter (`Owen-Wu-Yau-IEEE (1).docx`)

Open the .docx and replace the placeholders at the top:

| Field | Replace | With |
|---|---|---|
| Title (line 7 & 8) | `[TITLE - TO BE DECIDED]` | e.g. **"Echoes of Equity: Multi-Seed Evaluation of Fairness and Stability in Voice-Based Trustworthy-Intent Classification"** |
| Instructor Name | `[INSTRUCTOR NAME]` | Your actual mentor's name |
| Instructor Affiliation | `[INSTRUCTOR AFFILIATION]` | Your mentor's institution |

**Tip:** the title mirrors the abstract's two contributions ("multi-seed" + "fairness/stability"). Keep it under 15 words; ISEF prefers a single readable title.

## Step 1.2 — Fix Table 5 (the broken headline table)

In the docx, find Table 5 (Sec. IV-E). Currently it shows literal letters. Replace with the real numbers from `MySolution/Analysis/results.json`:

```
Table 5: Across-seed standard deviation of overall accuracy and fairness gap
        (5 seeds, same fixed split).

Model | Accuracy std | Fairness-gap std
RF    | 0.97 %       | 4.59 pp
LR    | 0.00 %       | 0.00 pp
ANN   | 2.53 %       | 1.10 pp
CNN   | 0.92 %       | 2.87 pp
DANN  | 0.88 %       | 0.89 pp
```

Then, immediately below the table, add one sentence pointing the judge at the punch line:

> "The two rows that matter are CNN and DANN: nearly identical mean fairness gap (4.97 vs 5.03 pp, Table 2), but a 3.2× narrower seed-to-seed spread for DANN (0.89 vs 2.87 pp)."

## Step 1.3 — Add the ablation table (Table 8)

**A. Run the ablation script.** From the project root:

```bash
cd MySolution/Analysis
python run_dann_ablation.py
```

This produces `MySolution/Analysis/ablation_results.json` and prints Table 8 to stdout.

**B. Add a new Section VI.G (or renumber as needed) to the paper**, right after Section V-B ("What does 'Fair' mean here?"). Draft prose:

> **G. Ablation Study**
>
> To isolate which component of DANN-Trust drives the stability advantage, we retrained four variants on the same fixed split and five seeds (Table 8). Removing the gradient-reversal layer (`no_grl`) reduces to a plain multi-task learner; removing the domain head entirely (`no_domain`) reduces to a plain ANN with the same encoder shape. If DANN's stability came from architectural regularization rather than the adversary, `no_grl` and `no_domain` would match the full model. They do not: [insert observed pattern from your run].

**C. Table skeleton** — fill numbers from the printed output:

```
Table 8: DANN-Trust ablation (mean ± std over 5 seeds; same fixed split).

Variant        | Accuracy       | AUC              | Δ_eth (pp)
dann_full      | XX.XX ± X.XX%  | 0.XXX ± 0.XXX    | X.XX ± X.XX
dann_no_grl    | XX.XX ± X.XX%  | 0.XXX ± 0.XXX    | X.XX ± X.XX
dann_no_domain | XX.XX ± X.XX%  | 0.XXX ± 0.XXX    | X.XX ± X.XX
dann_no_bn     | XX.XX ± X.XX%  | 0.XXX ± 0.XXX    | X.XX ± X.XX
dann_top15     | XX.XX ± X.XX%  | 0.XXX ± 0.XXX    | X.XX ± X.XX
```

Note in the caption: "`dann_no_domain` is architecturally identical to the ANN of Section III-D but with a 128→32 encoder rather than 128→64→32 — it is the correct control for the adversary."

## Step 1.4 — Unify figure numbering

Your `MySolution/Paper/figures/` folder has `fig01`..`fig06`, `fig09`, `fig13`. Fix the gaps.

**A. Rename files** so numbers are contiguous with what the paper refers to:

| Current filename | New filename | Referenced in paper as |
|---|---|---|
| `fig01_pipeline.png` | keep | Fig. 2 → change *paper* to say Fig. 1 |
| `fig02_demographics.png` | `fig02_demographics.png` | (currently no ref — add one in §II-A after Table 1) |
| `fig03_feature_boxplots.png` | keep | Fig. 1 → renumber to Fig. 3 |
| `fig04_ann_arch.png` | keep | Fig. 3 → renumber to Fig. 4 |
| `fig05_cnn_arch.png` | keep | Fig. 4 → renumber to Fig. 5 |
| `fig06_dann_arch.png` | keep | Fig. 5 → renumber to Fig. 6 |
| `fig09_acc_by_ethnicity.png` | `fig07_acc_by_ethnicity.png` | Fig. 6 → Fig. 7 |
| `fig13_fairness_pareto.png` | `fig08_fairness_pareto.png` | Fig. 7 → Fig. 8 |

Recommendation: **pick "the paper's numbering wins"** and rename the PNGs to match, so the folder is `fig01`..`fig08`, no gaps.

**B. Regenerate figures.** After renaming, run:

```bash
cd MySolution/Analysis
python make_figures.py
python make_arch_figures.py
```

**C. In the docx**, insert each figure by its new filename and re-check every "Fig. N" caption/reference.

---

# ITEM 2 — Methodological fixes (a couple of weekends)

## Step 2.1 — Switch to speaker-grouped split and rerun everything

**Why:** each speaker has 12 utterances. A pure `train_test_split` almost certainly puts the same speaker in both train and test, and the model can shortcut on speaker identity. This is the single most important defensibility fix.

### 2.1.a — Write a small helper `MySolution/Analysis/split_utils.py`

```python
# split_utils.py — one place that decides the train/test split for every model.
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

def grouped_stratified_split(speaker_ids, y_intent, y_ethn,
                             test_size=0.2, seed=42, n_tries=200):
    """
    GroupShuffleSplit by Speaker_ID (no speaker appears in both halves),
    while picking the draw that best preserves the joint intent x ethnicity
    proportions on the test side. Returns (train_idx, test_idx).
    """
    strata = (y_intent.astype(int) * 10 + y_ethn.astype(int))
    target = Counter(strata)
    target_prop = {k: v / len(strata) for k, v in target.items()}

    best_idx, best_score = None, np.inf
    gss = GroupShuffleSplit(n_splits=n_tries, test_size=test_size,
                            random_state=seed)
    for tr, te in gss.split(np.zeros(len(speaker_ids)),
                            groups=speaker_ids):
        te_counter = Counter(strata[te])
        te_prop = {k: te_counter.get(k, 0) / len(te) for k in target_prop}
        # L1 distance between test-set stratum proportions and corpus proportions
        score = sum(abs(te_prop[k] - target_prop[k]) for k in target_prop)
        if score < best_score:
            best_score, best_idx = score, (tr, te)
    return best_idx
```

### 2.1.b — Use it in every model script

Replace the current `train_test_split(...)` block. For example in `compare_all_models.py`:

```python
from split_utils import grouped_stratified_split

speaker_ids = df["Speaker_ID"].values
tr_idx, te_idx = grouped_stratified_split(
    speaker_ids, y_trust, y_ethn, test_size=0.2, seed=SPLIT_SEED
)
X_tr, X_te = X[tr_idx], X[te_idx]
yt_tr, yt_te = y_trust[tr_idx], y_trust[te_idx]
ye_tr, ye_te = y_ethn[tr_idx], y_ethn[te_idx]
es_tr, es_te = ethn_str[tr_idx], ethn_str[te_idx]
ag_tr, ag_te = age_str[tr_idx],  age_str[te_idx]
sx_tr, sx_te = sex_str[tr_idx],  sex_str[te_idx]
```

Do the same edit in:

- `ExistingMethods/LogisticRegression.py`
- `ExistingMethods/RandomForest.py`
- `NewMethods/ANN_Trustworthy_Intent_Project/ann_trust_model.py`
- `NewMethods/CNN_Trustworthy_Intent_Project/cnn_trust_model.py`
- `NewMethods/DANN_Trustworthy_Intent_Project/dann_trust_model.py`
- `Analysis/run_dann_ablation.py`

### 2.1.c — Also fix the imputation leak while you're in there

In every script, change:

```python
X = X.fillna(X.mean())         # BEFORE — fits on train+test
```

to:

```python
train_mean = X.iloc[tr_idx].replace([np.inf, -np.inf], np.nan).mean()
X = X.replace([np.inf, -np.inf], np.nan).fillna(train_mean)
```

### 2.1.d — Rerun and regenerate all tables/figures

```bash
cd MySolution/Analysis
python compare_all_models.py       # → new results.json
python run_dann_ablation.py        # → new ablation_results.json
python make_figures.py             # → regenerates fig07/fig08
```

### 2.1.e — Update the paper to reflect the new protocol

In §II-A.3 (Pre-processing), replace:

> "Finally, we split the corpus 80/20 into train (921 utterances) and held-out test (231 utterances) using train_test_split with random_state = 42 and a joint stratification key that combines intent and ethnicity..."

With:

> "Finally, we split the corpus 80/20 using `GroupShuffleSplit` on `Speaker_ID`, so no speaker appears in both train and test. Among 200 candidate draws under seed 42, we selected the split whose test-set joint intent × ethnicity proportions best matched the corpus. This yields ~77 speakers (924 utterances) in train and ~19 speakers (228 utterances) in test. Grouped splitting is essential for a fairness study because each speaker contributes 12 utterances; an ungrouped split allows the classifier to shortcut on speaker identity, inflating accuracy and confounding per-group performance."

Also add one sentence to the abstract: "We use a speaker-disjoint 80/20 split to prevent identity leakage."

Expect absolute accuracies to drop by ~3–8 pp. The **comparative** story (deep > baseline, DANN more stable) will almost certainly survive.

## Step 2.2 — Intersectional (age × ethnicity) breakdown

Your test set is small, so run this as a *supplementary* analysis rather than a headline.

### 2.2.a — Add code at the end of `compare_all_models.py`

```python
# --------------------------------------------------------------------------
# Intersectional slice: age x ethnicity (6 cells).  Cells with n<10 are
# reported but flagged as low-confidence.
# --------------------------------------------------------------------------
INTERSECT_ORDER = [(a, e) for a in AGE_ORDER for e in ETH_ORDER]

def _intersect_metrics(probs, preds):
    out = {}
    for age, eth in INTERSECT_ORDER:
        mask = (ag_te == age) & (es_te == eth)
        if mask.sum() < 2:
            continue
        try:
            g_auc = float(roc_auc_score(yt_te[mask], probs[mask]))
        except ValueError:
            g_auc = float("nan")
        out[f"{age}_{eth}"] = {
            "n": int(mask.sum()),
            "acc": float(accuracy_score(yt_te[mask], preds[mask])),
            "auc": g_auc,
        }
    return out

# Extend the per-model loop already in the file:
#   ... after existing per-seed evaluation ...
#   r["per_ageeth"] = _intersect_metrics(probs, preds)
```

### 2.2.b — Add Table 6 to the paper

```
Table 6: Intersectional accuracy (age x ethnicity), mean +/- std over 5 seeds.
         Cells with n<10 marked with *.

Model | Younger White | Younger Black | Younger S.Asian |
       Older White   | Older Black*  | Older S.Asian*
```

### 2.2.c — Add one paragraph of interpretation

Something like:

> "Table 6 disaggregates accuracy along the intersection of age and ethnicity. As Table 1 shows, the older Black (N=8) and older South Asian (N=8) cells are small, so per-seed accuracy is noisy — we mark them with an asterisk. Even so, [describe the pattern you see; e.g., 'the largest single-cell gap for RF is between Younger White and Older South Asian at XXpp, and this collapses to XXpp under DANN-Trust']. This intersectional pattern is not visible in the ethnicity-only breakdown of Table 3."

## Step 2.3 — Probe control on ANN and CNN encoders

**Why:** your paper reports DANN-encoder ethnicity probe = 53.7% but never shows what the *non-adversarial* encoders leak. Without that baseline, "20 pp above chance" is uninterpretable.

### 2.3.a — Create `MySolution/Analysis/run_probe_baselines.py`

```python
# Trains a fresh LR probe on frozen encoder features from ANN, CNN, and DANN,
# to measure how much ethnicity information each encoder retains.
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- import the shared data + model builders from compare_all_models.py ---
# (or copy the data-loading block; do NOT re-run all 5 models)

SEEDS = [42, 43, 44, 45, 46]
NUM_ETH = 3  # White, Black, South_Asian

def probe_from_encoder(build_and_train_fn, X_tr, X_te, y_ethn_tr, y_ethn_te):
    accs = []
    for s in SEEDS:
        np.random.seed(s); tf.random.set_seed(s)
        full_model, encoder_output_name = build_and_train_fn(s)
        encoder_only = Model(
            inputs=full_model.inputs,
            outputs=full_model.get_layer(encoder_output_name).output,
        )
        Z_tr = encoder_only.predict(X_tr, verbose=0)
        Z_te = encoder_only.predict(X_te, verbose=0)
        probe = LogisticRegression(max_iter=2000, random_state=s)
        probe.fit(Z_tr, y_ethn_tr)
        accs.append(accuracy_score(y_ethn_te, probe.predict(Z_te)))
    return float(np.mean(accs)), float(np.std(accs))

# For each of ANN / CNN / DANN, name the final encoder layer explicitly
# in the model builder (e.g. name="encoder_output") so we can grab it here.
```

### 2.3.b — Small change in the ANN/CNN builders

In `ann_trust_model.py`, name the last hidden layer:

```python
Dense(32, activation='relu', name='encoder_output'),
Dense(1, activation='sigmoid', name='trust_output')
```

In `cnn_trust_model.py`, name the `Dense(64)` layer:

```python
Dense(64, activation='relu', name='encoder_output'),
Dense(1, activation='sigmoid', name='trust_output')
```

### 2.3.c — Add Table 7 to the paper

```
Table 7: Ethnicity probe accuracy on frozen encoder outputs
         (mean +/- std over 5 seeds; chance = 33.3%).

Model | Probe acc  | Leakage above chance
ANN   | XX.X ± X.X | +XX.X pp
CNN   | XX.X ± X.X | +XX.X pp
DANN  | 53.7 ± X.X | +20.4 pp
```

### 2.3.d — Rewrite the last paragraph of §IV-E

Currently:

> "This probe reaches 53.7% mean accuracy across the five seeds, against a chance level of 33.3% ..."

Replace with:

> "Table 7 compares this probe across all three deep encoders. The non-adversarial ANN and CNN encoders retain [XX%] and [XX%] ethnicity information respectively; DANN-Trust's adversarial objective reduces this to 53.7%. The adversary therefore removes roughly [Y] of the recoverable ethnicity signal — a meaningful reduction, but not full invariance."

That reframing is much stronger: instead of "DANN still leaks 20 pp above chance" (which sounds like a failure), it becomes "DANN removes X of the leakage the non-adversarial encoders exhibit" (which is what your data actually shows).

---

# Verification checklist after all edits

Before you re-export the docx, run this pass:

- [ ] Title, authors, mentors filled in on page 1.
- [ ] Every `[TABLE]` referenced in the text has real numbers.
- [ ] `grep -n "TITLE" *.md` and `grep -n "TODO" *.md` return nothing in `MySolution/Paper/`.
- [ ] All Fig. N in the paper text match a real PNG in `MySolution/Paper/figures/`.
- [ ] `python compare_all_models.py` completes and `results.json` regenerates.
- [ ] `python run_dann_ablation.py` completes and `ablation_results.json` regenerates.
- [ ] `python run_probe_baselines.py` completes.
- [ ] The abstract's numbers match Table 2 and Table 5 after rerunning.
