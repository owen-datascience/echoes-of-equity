# VI. Discussion

This section interprets the experimental results of Section V. Section VI-A discusses what worked — the three distinct wins our experiments produced. Section VI-B examines the fairness–accuracy trade-off visually summarised in Fig. 13. Section VI-C lists the limitations of our study and the work needed to address each one.

## VI-A What Worked

Three distinct findings emerged from the experiments, each backed by the multi-seed numbers of Section V.

**Accuracy lift from depth.** Every deep model exceeded both linear baselines: the ANN gained 3.1 percentage points over Random Forest, the CNN 2.0pp, and DANN-Trust 1.6pp. The lift was consistent across all five training seeds, with deep-model accuracy standard deviation below 2.5% in every case. This confirms that the source paper's $\sim$71% plateau is a property of the linear/tree-based model class rather than a true ceiling imposed by the dataset itself: the 60-dimensional acoustic feature space contains non-linear interactions that linear classifiers cannot represent and that modestly sized neural networks can.

**Fairness lift from depth.** The non-linear capacity of deep models more than halved the per-ethnicity accuracy gap on this corpus: Random Forest's 12.07pp gap fell to 4.97pp under CNN and 5.03pp under DANN-Trust. The CNN and DANN gaps both *match* the source paper's published 5pp gap. The biggest beneficiary was the South Asian group, whose mean accuracy rose by approximately 8 percentage points absolute (66.6% under RF to 74.9% under CNN and 75.2% under DANN). This finding is the most surprising in the paper: it says that depth and non-linearity alone — without any fairness-specific architectural component — substantially close the per-ethnicity accuracy gap on the TIS Corpus.

**Reproducibility from adversarial training.** DANN-Trust matched CNN on mean accuracy (75.0% vs 75.4%) and on mean fairness gap (5.03pp vs 4.97pp), but the seed-to-seed standard deviation of both metrics dropped substantially: gap standard deviation from 2.87pp (CNN) to 0.89pp (DANN), accuracy standard deviation from 2.53% (ANN) to 0.88% (DANN). The adversarial objective therefore acts as a *regulariser on demographic shortcuts*: rather than each random initialisation discovering a different ethnicity-correlated shortcut and producing a different fairness gap, every DANN seed converges toward a similar ethnicity-suppressing solution.

**Acoustic findings replicate.** The Random Forest Gini importance ranking (Section V-G) confirmed the source paper's identification of F0 mean, F0 standard deviation, HNR, shimmer and CPP as the dominant trustworthy-intent acoustic cues, with the LTAS family contributing little. This independent replication on the same corpus, under a different evaluation split, increases our confidence in those acoustic findings.

## VI-B The Fairness–Accuracy Trade-off

Fig. 13 plots every model as a point in the (overall accuracy, fairness gap) plane, with $\pm 1\sigma$ error bars in both dimensions. The "good" corner is the lower-right (high accuracy, low gap), and the green arrow on the figure points in that direction. Random Forest and Logistic Regression occupy the upper-left region with high gaps and modest accuracy; ANN sits at the upper-right (highest accuracy but a gap still above 10pp); CNN and DANN-Trust occupy the lower-right.

The crucial visual difference between CNN and DANN-Trust is the *size of the error bars*. CNN's vertical (gap) error bar extends approximately ±3pp, meaning a CNN retrained on the same data with a different seed may land anywhere from a 2pp gap to an 8pp gap. DANN-Trust's vertical error bar is approximately ±1pp, confining the same uncertainty to a 4pp-to-6pp range. The two models therefore are *not* equivalent for deployment: CNN's "5pp average" promise is conditional on which seed was used, while DANN-Trust's same "5pp average" promise is robust to that choice. For any voice-AI deployment in which a model is retrained periodically — most production systems — the second guarantee is more useful than the first, even at the cost of giving up the 0.4pp of peak accuracy that CNN holds.

Fig. 13 also marks the source paper's Random Forest LOSO result as a black star at (71%, 5.0pp). All of our deep models lie to the right of this star (higher accuracy at comparable or lower gap), confirming that our headline contribution holds against the published baseline.

## VI-C Limitations

Table 10 summarises the six most important limitations of the present study and the work needed to address each. We highlight three in prose because they shape the interpretation of every result above.

First, the **per-cell sample sizes for older Black and older South Asian speakers are very small** (N=8 each, see Table 1), which limits the precision of any per-demographic accuracy estimate based on those groups. We mitigate this through joint stratification and seed averaging but cannot eliminate the underlying small-sample variance. Bootstrap 95% confidence intervals on per-group accuracy are a natural next step.

Second, the **trustworthy-intent label is self-rated**: speakers were asked to convey trust as *they* understood it, with no listener-perception validation. A speaker who believes they sound trustworthy may not in fact be perceived that way by a listener. The acoustic signal we are classifying is therefore a *production* signal, not a *perception* signal, and the two may diverge in ways that current data cannot reveal.

Third, the **DANN-Trust adversarial mechanism is only partially successful at scrubbing demographic information**: the probe classifier of Section V-I recovers ethnicity from the encoder at approximately 54% accuracy versus a chance of 33%. The encoder is therefore not demographic-invariant in the strict information-theoretic sense; what it achieves is the weaker but operationally relevant property of *not using* the residual demographic information to make per-group disparate predictions. Closing the remaining 20-percentage-point leakage may require a larger encoder bottleneck, a deeper adversarial head, or per-attribute multi-axis adversarial heads, all of which are concrete future-work directions.

**Table 10: Study limitations and proposed mitigations.**

| # | Limitation                                                                 | Proposed mitigation                                                                                  |
|--:|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 1 | Small N for older Black (N=8) and older South Asian (N=8) speakers         | Bootstrap 95% CIs on per-group accuracy; data collection of additional older non-white speakers      |
| 2 | Self-rated trustworthy intent (no listener-perception validation)          | Pair every utterance with crowd ratings; train production-to-perception alignment models             |
| 3 | Single dataset (no cross-corpus generalisation test)                       | Re-evaluate every model on a second voice-trust corpus once available (e.g., Mandarin or Latino)     |
| 4 | DANN $\lambda$ schedule not exhaustively grid-searched                     | Sweep $(\lambda_{\max}, \text{ramp shape})$ with held-out validation; report sensitivity curves      |
| 5 | Single 80/20 train/test split (multi-seed shares this split)               | Add 5-fold cross-validation across splits, on top of multi-seed across initialisations               |
| 6 | DANN encoder still leaks ethnicity (probe 54% vs 33% chance)               | Larger encoder bottleneck; deeper adversarial head; multi-axis adversarial (ethnicity + age + sex)   |

The first five limitations are addressed by additional data collection or computational budget. The sixth is methodological and is the most interesting research direction the present work opens up: if DANN-Trust's stability win persists when the leakage drops to near-chance, the case for adversarial debiasing in voice AI becomes considerably stronger.
