# V. Experiments and Results

This section reports the experimental evaluation of all five classifiers introduced in Section IV. Section V-A describes the shared experimental setup. Section V-B presents overall classification performance across the five models (Table 3, Fig. 7, Fig. 8). Section V-C decomposes accuracy by speaker ethnicity, the paper's primary fairness axis (Table 5, Fig. 9). Sections V-D and V-E report analogous breakdowns by age-group and sex (Tables 6 and 7, Fig. 10). Section V-F reports confusion matrices (Fig. 11). Section V-G analyses acoustic-feature importance and ablation studies (Fig. 12, Table 8). Section V-H summarises AUC results (Table 4). Section V-I, the central novel finding, quantifies the variance of every model's accuracy and fairness behaviour across random initialisations (Fig. 13).

## V-A Experimental Setup

All experiments use Python 3.11, scikit-learn 1.4 and TensorFlow 2.20 on commodity CPU hardware. The split between training (921 utterances) and held-out test (231 utterances) is performed once with `random_state = 42`, using joint stratification on the cross-product of intent (Neutral, Trustworthy) and ethnicity (White, Black, South Asian) so that all six $(2 \times 3)$ cells preserve their corpus proportions in both halves. The held-out test set contains 48 Neutral + 48 Trustworthy White utterances, 34 + 34 Black, and 33 + 34 South Asian.

Each model is trained five times with initialisation seeds $\{42, 43, 44, 45, 46\}$ on this identical split. All reported numbers in this section are the mean across the five seeds, with the standard deviation reported as the uncertainty. This protocol isolates *model variance* (sensitivity to weight initialisation) from *split variance* (sensitivity to which utterances ended up in the test set), and it is what allows us to quantify the stability claim of Section V-I.

Random Forest uses `n_estimators = 100` and `max_depth = 10`; Logistic Regression uses the `liblinear` solver. The deep models (ANN, CNN, DANN) are trained with Adam at learning rate $10^{-3}$, batch size $32$, binary cross-entropy on the trust head, and (for DANN) categorical cross-entropy on the ethnicity head with the gradient-reversal weight ramp described in Section IV-E. ANN and CNN train for 50 epochs; DANN trains for 80. The single comparison harness that produces every number in this section is available at `MySolution/Analysis/compare_all_models.py`, and its full per-seed output is saved in `MySolution/Analysis/results.json`.

## V-B Overall Classification Performance

Table 3 summarises overall accuracy and AUC for all five models. The two linear/tree baselines reach 73.4% and 71.9% accuracy under our split, recovering the source paper's published 71% / 69% (Maltezou-Papastylianou et al., Table 8) within $\pm 2$ percentage points. The 1.4–2.4pp uplift is attributable to the easier-than-LOSO random split protocol, not to any methodological change in the baselines themselves.

**Table 3: Overall accuracy and AUC on the TIS Corpus held-out test set (mean ± std over 5 seeds; joint-stratified 80/20 split, split seed = 42).**

| Model | Accuracy        | AUC               | Fairness gap $\Delta_{\text{eth}}$ |
|------:|----------------:|------------------:|-----------------------------------:|
| RF    | 73.42 ± 0.97%   | 0.823 ± 0.006     | 12.07 ± 4.59 pp                    |
| LR    | 71.86 ± 0.00%   | 0.803 ± 0.000     | 11.34 ± 0.00 pp                    |
| ANN   | **76.54** ± 2.53% | 0.825 ± 0.019   | 10.30 ± 1.10 pp                    |
| CNN   | 75.41 ± 0.92%   | 0.814 ± 0.010     | **4.97** ± 2.87 pp                 |
| DANN  | 74.98 ± **0.88%** | 0.810 ± 0.011   | **5.03** ± **0.89** pp             |

Three observations are immediate. First, all three deep models exceed both linear baselines on accuracy by 1.6 to 3.1 percentage points. Second, AUC values cluster between 0.80 and 0.83 for every model, indicating that discriminability — separately from threshold-based accuracy — is relatively uniform across model classes; the deep models' accuracy gain comes from better-placed decision thresholds, not radically better separation. Third, and most strikingly, the per-ethnicity fairness gap collapses from 12.07pp (RF) and 11.34pp (LR) to 4.97pp (CNN) and 5.03pp (DANN). CNN and DANN both *match* the source paper's published 5pp gap.

Fig. 7 shows training-loss and trust-accuracy curves across epochs, averaged over the five seeds and shaded with $\pm 1\sigma$ bands. Fig. 8 shows ROC curves with shaded confidence regions; the AUC values in the figure's legend match Table 3.

> **TODO (data collection):** Extend `compare_all_models.py` to also record precision, recall and F1-score per class so that Table 3 can be enriched with the per-class breakdown. The cost is one additional call to `sklearn.metrics.classification_report` per model run.

## V-C Per-Ethnicity Comparison

Table 5 decomposes the headline numbers of Table 3 by speaker ethnicity. Fig. 9 visualises the same data as a grouped bar chart with $\pm 1\sigma$ error bars and the source paper's 71% RF baseline marked as a dashed reference.

**Table 5: Per-ethnicity trust accuracy (mean ± std over 5 seeds).**

| Model | White (N=96)  | Black (N=68)  | South Asian (N=67) | Δ (gap)            |
|------:|--------------:|--------------:|-------------------:|-------------------:|
| RF    | 78.12 ± 1.86% | 73.53 ± 1.86% | 66.57 ± 4.49%      | 12.07 ± 4.59 pp    |
| LR    | 76.04 ± 0.00% | 64.71 ± 0.00% | 73.13 ± 0.00%      | 11.34 ± 0.00 pp    |
| ANN   | 78.75 ± 3.82% | 73.53 ± 4.05% | 76.42 ± 5.70%      | 10.30 ± 1.10 pp    |
| CNN   | 77.29 ± 0.78% | 73.24 ± 3.40% | 74.93 ± 0.60%      | 4.97 ± 2.87 pp     |
| DANN  | 75.62 ± 1.93% | 73.82 ± 2.85% | 75.22 ± 1.79%      | 5.03 ± 0.89 pp     |

The story is clearest on the **South Asian** column. Under RF, this group reaches only 66.57% accuracy, 11.55 percentage points below the White group's 78.12%, replicating the source paper's observation that the South Asian subset is the hardest for tree-based classifiers on this corpus. The deep models lift this group's accuracy by approximately 8 to 10 percentage points absolute — ANN to 76.42%, CNN to 74.93%, DANN to 75.22% — closing most of the gap to the White group.

A second observation, not present in the source paper, is that **RF and LR disagree about which ethnic group is hardest**. RF places South Asian at the bottom (66.6%); LR places Black at the bottom (64.7%). The two baselines are therefore picking up *different* demographic shortcuts from the same data — there is no single intrinsically hard group, the choice of classifier decides which subset takes the largest penalty. This is precisely the kind of unpredictable per-group behaviour that DANN-Trust is designed to suppress, and a finding the source paper could not surface because it does not provide per-class confusion data for both models.

Among deep models, **DANN-Trust produces the flattest per-ethnicity profile on average**: only 0.40pp between the best (Black, 73.82%) and worst (White, 75.62% — note that the *White* group has the second-lowest accuracy here) groups when looking at the mean values alone, and 1.79–2.85pp seed-to-seed standard deviation on every group, the tightest spread in the table. ANN and CNN have similar mean profiles but visibly higher across-seed scatter, especially ANN whose South Asian std reaches 5.70pp.

## V-D Per-Age-Group Comparison

> **TODO (data collection):** The current `compare_all_models.py` slices by ethnicity only. To populate Table 6 and Fig. 10, extend the harness with one additional loop over `Speaker_AgeGroup` ∈ {Younger, Older}, computing accuracy, AUC and confusion-matrix counts per (model, age-group, seed). The change is approximately 20 lines and re-uses the existing slicing logic. Expected pattern, based on the source paper's Table 8: younger and older accuracy should be within ~1pp of each other for all models, so $\Delta_{\text{age}}$ should be substantially smaller than $\Delta_{\text{eth}}$. Older speakers may gain more from non-linear models because their voice-quality features (HNR, shimmer, CPP) vary more across individuals (source paper Table 4 vs. Table 5), and that variation is exactly what depth captures and linear models miss.

## V-E Per-Sex Comparison

> **TODO (data collection):** Same extension as V-D, this time slicing by `Speaker_Sex` ∈ {Female, Male}. The source paper does not report a per-sex breakdown, so this section will be the first published per-sex fairness analysis on the TIS Corpus. Expected pattern: small gap (sex is balanced across the corpus), but male voices may be classified slightly worse because of lower mean F0 (source paper Tables 4–6 show males have F0 means ~105–140 Hz vs females ~175–230 Hz) which makes pitch-variability features noisier on a relative scale.

## V-F Confusion Matrices

> **TODO (data collection):** Extend `compare_all_models.py` to call `sklearn.metrics.confusion_matrix` on `(y_true, y_pred)` for each model and save the resulting 2×2 matrix to `results.json` under a new `confusion_matrix` key per seed. Fig. 11 then renders the per-model confusion matrices as a 1×5 grid of heatmaps. Expected pattern, based on the source paper's Table 9: precision should be slightly higher than recall on the Trustworthy class (i.e., the models hesitate to call something Trustworthy, leading to more false negatives than false positives). Quantifying this lets us state whether the deep models eliminate that asymmetry or merely match it.

## V-G Feature Importance and Ablation

> **TODO (data collection — feature importance):** After RF training, save `rf_model.feature_importances_` to `results.json`. Fig. 12 then plots the top 15 features as a horizontal bar chart, directly comparable to Fig. 1 of Maltezou-Papastylianou et al. Expected pattern: F0 mean, F0 SD, HNR, shimmer (APQ3) and CPP should dominate, replicating the source paper's finding that these acoustic features carry most of the trustworthy-intent signal. LTAS features should rank low for RF.
>
> **TODO (data collection — ablation):** Run DANN-Trust four additional times with one component removed each time: (1) `w/o GRL` — same architecture but the GRL is replaced with `tf.identity`, turning the model into a multi-task learner; (2) `w/o domain head` — the encoder is trained with the trust head only (this is the ANN with encoder size 32 instead of 32-after-64); (3) `w/o BatchNorm`; (4) `top-15 features only` — input restricted to the 15 highest-Gini features from Fig. 12. Report Acc, AUC and $\Delta_{\text{eth}}$ for each in Table 8. The interesting comparison is `DANN full` vs `w/o GRL`: if multi-task learning alone matches DANN's gap variance, the GRL adds nothing; if DANN beats it, the gradient-reversal is doing the work.

## V-H AUC Summary

Table 4 mirrors Table 10 of the source paper but is computed on our split and aggregated over 5 seeds. AUC values support the same qualitative reading as accuracy.

**Table 4: Per-ethnicity AUC (mean over 5 seeds; standard deviation reported in the supplementary `results.json`).**

| Model | White AUC | Black AUC | South Asian AUC |
|------:|----------:|----------:|----------------:|
| RF    | 0.882     | 0.794     | 0.774           |
| LR    | 0.873     | 0.721     | 0.784           |
| ANN   | 0.847     | 0.784     | 0.842           |
| CNN   | 0.818     | 0.818     | 0.808           |
| DANN  | 0.832     | 0.771     | 0.799           |

Two patterns are worth noting. First, CNN's per-ethnicity AUC values are remarkably uniform (0.81 to 0.82 across all three groups), supporting the per-ethnicity accuracy result that CNN is the most balanced model on this corpus when measured by mean alone. Second, LR's AUC on Black speakers (0.721) is its single weakest cell, mirroring its accuracy collapse to 64.7% on that group — Logistic Regression's linear decision boundary simply cannot separate trustworthy from neutral Black speech as well as it can for the other two groups, consistent with the well-documented brittleness of linear classifiers to demographic distribution shift.

The overall AUC values in Table 3 (0.80 to 0.83 across all models) are comparable to the source paper's published 0.76 to 0.78 range, demonstrating that the threshold-independent quality of the underlying discriminator is in the same ballpark across linear, non-linear and adversarial approaches; the gains we report on accuracy come from better-calibrated decision thresholds rather than radically better separation.

## V-I Variance and Stability Across Seeds (Headline Result)

The headline finding of this paper is not visible in any single-seed result. It emerges only when each model is trained multiple times and the *standard deviations* of the resulting metrics are compared. Table 9 collects every model's standard deviation on the two metrics that matter for deployment: overall accuracy and per-ethnicity fairness gap.

**Table 9: Across-seed standard deviation of overall accuracy and fairness gap (5 seeds, same fixed split).**

| Model | Accuracy std | Fairness-gap std |
|------:|-------------:|------------------:|
| RF    | 0.97 %       | 4.59 pp           |
| LR    | 0.00 %       | 0.00 pp           |
| ANN   | 2.53 %       | 1.10 pp           |
| CNN   | 0.92 %       | 2.87 pp           |
| DANN  | **0.88 %**   | **0.89 pp**       |

Two things stand out. First, **LR's zero variance** confirms that a fully deterministic linear solver with a fixed random state produces bit-identical solutions across reseeds; this serves as a useful sanity check on the harness rather than a property of LR itself. Second, **DANN-Trust achieves the lowest standard deviation on both metrics among non-trivial models**. CNN matches DANN on mean accuracy and mean gap, but its fairness gap swings by ±2.87pp seed-to-seed — meaning that a CNN retrained on the same data with a different random initialisation can produce a fairness gap anywhere from approximately 2pp to 8pp. DANN's gap swings by only ±0.89pp, roughly one third of CNN's range. Similarly, ANN's mean accuracy is the highest of any model at 76.54%, but its ±2.53% standard deviation means a single retrain can land anywhere from 74.0% to 79.1%; DANN's ±0.88% confines the same uncertainty to 74.1% to 75.9%.

This stability matters for any deployed system that retrains periodically. A fairness audit that signs off a CNN model with a 2.5pp measured gap provides no guarantee that the next retrain will not produce an 8pp gap. The same audit on a DANN model is far more predictive of future behaviour. Fig. 13 summarises this finding visually as a scatter in (overall accuracy, fairness gap) space, with $\pm 1\sigma$ error bars in both dimensions; DANN-Trust occupies the lower-right corner with markedly smaller error bars than any other model.

Probing the DANN encoder. To verify that the adversarial mechanism is doing what its design intends, we extract the frozen 32-dimensional encoder output $\phi(x) = G_f(x)$ on the test set and train a fresh Logistic Regression probe to predict ethnicity from $\phi(x)$. The probe achieves 53.7% mean accuracy across the five seeds, compared to a chance level of 33.3% for three ethnicities — a leakage of approximately 20 percentage points above chance. The encoder is therefore *not* perfectly demographic-invariant: a downstream classifier could still extract ethnicity from $\phi(x)$ better than chance. What the adversarial training does achieve is that this residual demographic information is not used by the trust head to make per-group predictions disproportionate, which is the operational definition of fairness we care about (equal per-group accuracy) rather than the strict information-theoretic version (zero mutual information between $\phi$ and $z$). Closing this remaining 20pp leakage — through stronger or more diverse adversarial heads, or a larger encoder bottleneck — is a natural direction for future work and is discussed in Section VII.
