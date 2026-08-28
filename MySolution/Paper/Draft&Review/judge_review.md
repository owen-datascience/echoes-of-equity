# ISEF-Level Judge Review — "Echoes of Equity" (Owen Wu)

## Overall Verdict

This is a **strong, well-motivated project** with a genuinely interesting central finding (fairness-metric variance across retrains). The writing is unusually mature for a high-school entrant, the code is clean and comment-rich, and the research question — that most of the demographic gap is a *model-class* artifact rather than a data artifact — is a real, defensible contribution. However, several **methodological and presentation issues** would cost you points at ISEF / a top venue and are all fixable with modest effort.

---

## 1. Show-stopping issues (fix before submission)

### 1.1 The paper's front matter and headline table are broken

- Lines 7–8 of the docx: **`[TITLE - TO BE DECIDED]`** and lines 5–6 `[INSTRUCTOR NAME]`, `[INSTRUCTOR AFFILIATION]` — these placeholders are still in the document. A judge reading the first page will assume the paper is unfinished.
- **Table 5** (Sec. IV-E, the paper's self-described "headline result") is unreadable — the fairness-gap std column literally says `F. pp / A. pp / F. pp / D. pp / B. pp` instead of numbers. From `results.json` the correct values are `4.59 / 0.00 / 1.10 / 2.87 / 0.89 pp`. Fix this table — it *is* your thesis.

### 1.2 Almost certain speaker leakage in the train/test split — this is a fairness paper's worst nightmare

Every speaker contributes **12 utterances** (6 sentences × 2 intents). Your split uses `train_test_split(..., stratify=intent×ethnicity)` at the **utterance** level. Nothing groups by `Speaker_ID`, so the same speaker almost certainly appears in both train and test. This means:

- The classifier can learn "this is speaker #47's voice" and use identity as a shortcut.
- Your 73–77% accuracies are inflated relative to what a genuinely unseen-speaker deployment would produce (which is why the source paper's LOSO gives 71%/69% — the honest comparison).
- More importantly for a **fairness** paper: per-group accuracy differences may partly reflect who happened to end up in which side of the split.

**Fix:** switch to `sklearn.model_selection.GroupShuffleSplit` (or `GroupKFold`) on `Speaker_ID`, with stratification approximated by re-drawing until intent×ethnicity balance is acceptable. Rerun everything. Numbers will drop but the *comparative story* (deep > baseline, DANN more stable) is very likely to survive, and the paper will be defensible.

### 1.3 "Stability" is only measured over 5 seeds and one split

- Section V-C admits this, but the paper's central claim ("CNN 2.87pp std vs DANN 0.89pp std → DANN is more stable") rests on **five samples per model**. That is not a defensible variance estimate — an F-test on 5-vs-5 with those numbers is borderline significant at best, and you don't report one.
- The right story is either (a) run 25+ seeds so a variance ratio test has teeth, or (b) do a **bootstrap** over per-seed gap values and report 95% CIs, or (c) run **5 seeds × 5 splits = 25 runs** as the limitation section already promises.

### 1.4 The ablation study is in the code but not in the paper

`run_dann_ablation.py` has four sensible variants (`no_grl`, `no_domain`, `no_bn`, `top15`). **None of them appear in the paper**, and there is no Table 8, despite the script writing one. This is a wasted contribution. Include the ablation because it directly answers the question a skeptical judge will ask:

> *"Is the fairness gain from the GRL, or just from the shallower encoder / dropout / batchnorm regularizing the model?"*

Without `no_grl` and `no_domain` numbers, you cannot claim the adversary is doing the work.

---

## 2. Serious methodological gaps

### 2.1 CNN vs DANN comparison isn't apples-to-apples

The DANN encoder is **128 → 32** but your ANN is **128 → 64 → 32** (`dann_trust_model.py:171-175` vs `ann_trust_model.py:49-58`). You even note "The middle layer was removed to reduce parameter count under adversarial pressure." That's a legitimate design choice, but it means any accuracy/stability delta between ANN and DANN could be encoder capacity, not adversarial training. The `dann_no_domain` ablation (encoder + trust head only, same 128→32) is exactly the right control — report it.

### 2.2 Higher accuracy but lower AUC is a red flag

Your deep models beat RF on accuracy (75–77% vs 73%) but have **lower or equal AUC** (0.81 vs 0.82). You explain this away as "better-calibrated thresholds," but a judge will read that as: *the neural nets are not actually more discriminative; they've just landed a slightly better default 0.5 cut on this particular test set*. Show a threshold sweep, F1 at the operating point, or Youden's J. If the accuracy advantage evaporates under a fairer threshold, that changes the story.

### 2.3 The probe result undercuts the framing more than you acknowledge

53.7% probe accuracy vs 33.3% chance = **~20 pp of leakage above chance**. That is a *lot* — the encoder has plainly not achieved anything close to demographic invariance. Section V-B handles this reasonably ("reduced, but did not eliminate") but the abstract still calls it a "sign of residual demographic leakage" without quantifying that the adversary has removed maybe half of the recoverable information (baseline probe on the raw features would clarify this). **Run and report a probe on the ANN/CNN encoder** as a control — if their probes hit ~65%, you can honestly say "DANN reduced leakage from 65% to 54%". Without that baseline, the number is uninterpretable.

### 2.4 Intersectional analysis is missing

Your code computes per-age and per-sex slices (`compare_all_models.py:263-282`) but only ethnicity is in the paper. Given that Table 1 shows very small older Black / older South Asian cells (N=8 each), the most likely place fairness fails is the **intersection**. Even a single paragraph reporting age × ethnicity accuracy (or the smallest cell you can support) would strengthen the paper significantly — and could be more of a headline than DANN itself.

---

## 3. Presentation and paper-craft improvements

- **Tie the intent label vs perceived-trust caveat harder.** The abstract implies suitability for hiring/loans/clinical triage; §II-C admits the model classifies *intent*, not *perceived* trust. Judges will notice the tension. Add one sentence in the conclusion explicitly saying *this system should not be deployed as a hiring aid* — that ethical restraint elevates the paper.
- **Section V-B header:** "What does 'Fair' means here?" → "What does 'fair' mean here?"
- **Consistency on capitalization:** you write both "black speakers" and "Black speakers" — pick one (Black, capitalized, is the current AP/academic convention).
- **Figures:** the paper mentions Fig. 7 (fairness-vs-accuracy) but the folder contains `fig13_fairness_pareto.png`. Rename consistently and confirm every figure referenced in the text is actually inserted.
- **References missing recent work:** given this is a 2025 fairness-in-speech paper, adding Feng et al. "Quantifying Bias in ASR" (2024), Meyer et al. "Artie Bias Corpus" (2020), and a citation to Zhao et al. "Men Also Like Shopping" for the general form of the leakage-shortcut argument would round out the related work.
- **Reproducibility:** ship a `requirements.txt` (TF 2.20, sklearn 1.4, python 3.11 pinned) and a `README.md` at the repo root that maps each script to the tables/figures it produces. Judges may not run your code, but the *presence* of a reproducibility bundle is a strong quality signal.
- **Number of decimal places:** you report gaps to two decimals (e.g., "4.97pp"). At N≈67-96 per cell, this is precision the sample size doesn't support — round to one decimal or add a "±" everywhere.

---

## 4. Smaller code-level notes (worth cleaning)

- `dann_trust_model.py:80-93`: the lambda schedule uses `p = epoch / total_epochs`, so the last epoch has `p = (N-1)/N` and λ ≈ 0.99, never quite reaching `lambda_max`. Minor, but if you say "λ reaches 1.0" in the paper, use `p = (epoch + 1) / total_epochs`.
- Same file, line 125: `X.fillna(X.mean())` is fit on the **full** X (train + test). That's a mild leakage — fit the imputer on train only. The `LR/RF/ANN/CNN` scripts have the same issue.
- Same for `LabelEncoder` on ethnicity in the `LR/RF/ANN/CNN` scripts — encoding across the full dataframe is harmless here (label set is fixed) but standardize the pattern.
- LR's std being exactly 0.00 across seeds is correct for a deterministic solver, but it's worth **noting once in the caption** rather than three times in the running text.

---

## 5. What is genuinely strong (keep doing this)

- The **framing** — "much of the disparity is model, not data" — is a real, publishable finding, novel to this corpus.
- The **multi-seed protocol** is exactly the right response to the D'Amour / Qian critique. The methodology section shows genuine understanding.
- The **probe evaluation** is a real technical contribution most high-school papers don't attempt.
- The **code quality** — comments in `dann_trust_model.py` explaining the GRL, λ schedule, and stratification decisions — reads like the author actually understands the model rather than pattern-matching a tutorial. That will show in judge Q&A.
- Frank limitations section and acknowledgement of what's left for future work.

---

## Suggested prioritization for revision

1. **This week:** fix Table 5, fill in title/instructor, add ablation table, unify Fig. numbering.
2. **This month:** switch to speaker-grouped split and rerun everything; add intersectional (age × ethnicity) breakdown; add probe control on ANN/CNN.
3. **Before ISEF finals:** scale to 25+ seeds and add a formal variance/F-test; add threshold-sweep analysis; add baseline-encoder probe.

If you land items 1 and 2, the paper is competitive at ISEF category level. If you land item 3 as well, the stability finding graduates from "interesting observation" to a defensible headline claim.
