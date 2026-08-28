"""
One-shot script: update Owen-Wu-Yau-IEEE (1)-Reviewed by Duo Chen.docx
with the 50-seed results and the no-BN DANN architecture story.
Backup already saved to Owen-Wu-Yau-IEEE (1)-Reviewed by Duo Chen-BACKUP-preUpdate.docx.
"""
from copy import deepcopy
from docx import Document

SRC = "Owen-Wu-Yau-IEEE (1)-Reviewed by Duo Chen.docx"

def set_cell_text(cell, text, bold=False):
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r.text = ""
    if p.runs:
        run = p.runs[0]
        run.text = text
    else:
        run = p.add_run(text)
    run.bold = bold
    for extra in cell.paragraphs[1:]:
        el = extra._element
        el.getparent().remove(el)

def replace_paragraph_text(paragraph, new_text):
    for r in list(paragraph.runs):
        r._element.getparent().remove(r._element)
    paragraph.add_run(new_text)

def insert_paragraph_after(paragraph, text, style=None):
    from docx.text.paragraph import Paragraph
    new_p = deepcopy(paragraph._p)
    for r in new_p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r'):
        r.getparent().remove(r)
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style is not None:
        try:
            new_para.style = style
        except KeyError:
            pass
    new_para.add_run(text)
    return new_para

d = Document(SRC)

# ---------- TABLES ----------
# Table 2 (doc index 1)
t = d.tables[1]
for i, row_vals in enumerate([
    ("RF",   "73.05 ± 1.27%", "0.818 ± 0.007", "12.29 ± 3.02 pp"),
    ("LR",   "71.86 ± 0.00%", "0.803 ± 0.000", "11.34 ± 0.00 pp"),
    ("ANN",  "76.25 ± 1.60%", "0.821 ± 0.009",  "7.99 ± 3.60 pp"),
    ("CNN",  "76.15 ± 1.75%", "0.828 ± 0.010",  "5.78 ± 2.73 pp"),
    ("DANN", "76.21 ± 1.71%", "0.828 ± 0.011",  "6.34 ± 3.14 pp"),
], start=1):
    for j, v in enumerate(row_vals):
        set_cell_text(t.rows[i].cells[j], v)

# Table 3 (doc index 2)
t = d.tables[2]
for i, row_vals in enumerate([
    ("RF",   "78.71 ± 1.76%", "71.35 ± 2.72%", "66.66 ± 2.87%", "12.29 ± 3.02 pp"),
    ("LR",   "76.04 ± 0.00%", "64.71 ± 0.00%", "73.13 ± 0.00%", "11.34 ± 0.00 pp"),
    ("ANN",  "79.25 ± 3.00%", "73.18 ± 2.84%", "75.07 ± 3.48%",  "7.99 ± 3.60 pp"),
    ("CNN",  "77.12 ± 2.12%", "74.79 ± 3.73%", "76.12 ± 3.49%",  "5.78 ± 2.73 pp"),
    ("DANN", "78.46 ± 2.67%", "74.24 ± 2.92%", "74.99 ± 3.15%",  "6.34 ± 3.14 pp"),
], start=1):
    for j, v in enumerate(row_vals):
        set_cell_text(t.rows[i].cells[j], v)

# Table 4 (doc index 3)
t = d.tables[3]
for i, row_vals in enumerate([
    ("RF",   "0.889 ± 0.011", "0.777 ± 0.020", "0.754 ± 0.018", "0.135"),
    ("LR",   "0.873 ± 0.000", "0.721 ± 0.000", "0.784 ± 0.000", "0.152"),
    ("ANN",  "0.861 ± 0.022", "0.782 ± 0.019", "0.807 ± 0.024", "0.079"),
    ("CNN",  "0.845 ± 0.013", "0.790 ± 0.022", "0.845 ± 0.021", "0.055"),
    ("DANN", "0.861 ± 0.013", "0.796 ± 0.021", "0.817 ± 0.022", "0.065"),
], start=1):
    for j, v in enumerate(row_vals):
        set_cell_text(t.rows[i].cells[j], v)

# Table 5 (doc index 4): add gap-std column
t = d.tables[4]
set_cell_text(t.rows[0].cells[0], "Model", bold=True)
set_cell_text(t.rows[0].cells[1], "Accuracy std", bold=True)
set_cell_text(t.rows[0].cells[2], "Δ_eth std (pp)", bold=True)
for i, row_vals in enumerate([
    ("RF",   "1.27%", "3.02"),
    ("LR",   "0.00%", "0.00"),
    ("ANN",  "1.60%", "3.60"),
    ("CNN",  "1.75%", "2.73"),
    ("DANN", "1.71%", "3.14"),
], start=1):
    for j, v in enumerate(row_vals):
        set_cell_text(t.rows[i].cells[j], v)

# ---------- PARAGRAPHS ----------
NEW_ABSTRACT = (
    "Abstract—The growing use of speech systems in consequential decisions, including hiring "
    "screens, loan applications, and clinical triage, means such systems demand fairness. Yet "
    "fairness is usually reported from a single training run, leaving the seed-to-seed variance "
    "of the fairness metric itself unmeasured. This paper presents a 50-seed evaluation of "
    "trustworthy-intent classification from voice on the Trustworthy Intent in Speech (TIS) "
    "Corpus of 1,152 utterances from 96 White, Black, and South Asian speakers. Five classifiers "
    "are retrained under 50 random seeds on a fixed joint-stratified 80/20 partition: Random "
    "Forest and Logistic Regression baselines from the source study, a feedforward network "
    "(ANN), a one-dimensional convolutional network (CNN), and DANN-Trust, which penalizes "
    "ethnicity-predictive information in the shared representation through a gradient reversal "
    "layer. The maximum-minus-minimum per-ethnicity accuracy gap Δ_eth falls from 12.29 "
    "percentage points (pp) under Random Forest to 5.78 pp under the CNN and 6.34 pp under "
    "DANN-Trust, indicating that much of the disparity is attributable to the model class rather "
    "than to the corpus. Pairwise Mann-Whitney U tests over 50 seeds confirm that CNN and "
    "DANN-Trust are statistically indistinguishable on accuracy (p=0.86), ethnicity gap (p=0.26), "
    "and sex gap (p=0.73); they are also indistinguishable on the stability of these metrics "
    "(Levene's p>0.30 for every pairing). An ablation on the DANN encoder further shows that "
    "removing BatchNormalization is required for the adversarial signal to help (accuracy "
    "p=0.002, gap mean p=0.011, gap variance p=0.049). A separate finding is that CNN and DANN "
    "both significantly reduce the female-vs-male accuracy gap relative to Random Forest "
    "(p=0.003 and p=0.002), an axis the corpus was not originally designed to control. A "
    "logistic-regression probe trained on the frozen DANN encoder still recovers ethnicity at "
    "~53% against a 33.3% chance baseline, so ethnicity information is reduced in influence on "
    "the trust head but not erased. The results support reporting fairness as a distribution "
    "over independent runs and treating architectural choices (e.g., normalization layers) as "
    "first-order fairness variables."
)
replace_paragraph_text(d.paragraphs[9], NEW_ABSTRACT)

NEW_INDEX = (
    "Index Terms—Algorithmic Fairness, Speech Processing, Domain-Adversarial Training, "
    "Acoustic Features, Model Stability, Trustworthiness, Multi-seed Evaluation, "
    "Ablation Study, Batch Normalization."
)
replace_paragraph_text(d.paragraphs[10], NEW_INDEX)

NEW_DANN_DESC = (
    "While ANN and CNN improve overall accuracy, their encoders can begin to use ethnicity-"
    "linked acoustic features to predict trustworthiness, producing per-group performance "
    "differences even in the absence of any explicit demographic input. To reduce this reliance, "
    "we introduce DANN-Trust, a Domain-Adversarial Neural Network [26]. The architecture "
    "consists of a shared encoder feeding two heads: a trust head that predicts the binary "
    "label, and a domain (ethnicity) head connected through a Gradient Reversal Layer (GRL). "
    "During backpropagation the GRL multiplies the domain-head gradient by -λ, so the encoder "
    "is trained to remove information the domain head could use, while the trust head is trained "
    "normally. Importantly, the encoder in DANN-Trust omits BatchNormalization: the ablation in "
    "Section IV-G shows that including BN with adversarial training significantly harms both "
    "trust accuracy (Mann-Whitney U, p=0.002) and the ethnicity gap (p=0.011), and inflates "
    "gap variance (Levene's, p=0.049). We interpret this as running batch statistics being "
    "unstable when the encoder's representation is being continuously reshaped by the "
    "adversarial signal on a small (n≈921 training) tabular dataset."
)
replace_paragraph_text(d.paragraphs[115], NEW_DANN_DESC)

NEW_G_SEEDS = (
    "Each model is trained 50 times with seeds {42, 43, ..., 91} on the same fixed train/test "
    "partition (split seed = 42). The partition is fixed for two reasons. First, to prevent "
    "data contamination, feature standardization is fit on the training split only and then "
    "applied to the test split. Second, holding the split fixed isolates model variance, the "
    "shift in a retrained model's behavior on identical data, from split variance, the "
    "contribution of the partition itself. Measuring seed-to-seed variance therefore allows "
    "us to test with statistical rigour whether DANN-Trust is not only fairer on average than "
    "the baselines but also whether it is more consistent across retraining than the CNN."
)
replace_paragraph_text(d.paragraphs[128], NEW_G_SEEDS)

NEW_A_SEEDS = (
    "Each model is trained 50 times with initialization seeds {42, 43, ..., 91} on an identical "
    "split, and all reported numbers are the mean across the 50 seeds with the standard "
    "deviation reported as uncertainty. Pairwise comparisons between models use the "
    "Mann-Whitney U test on the seed-level metric distributions (means), and Levene's test on "
    "their spreads (variances). This protocol isolates model variance from split variance and "
    "is what allows the stability claims of Section IV-E to be tested statistically."
)
replace_paragraph_text(d.paragraphs[139], NEW_A_SEEDS)

NEW_B_NARR = (
    "All three deep models exceed both baselines on overall accuracy by 4.29 to 4.39 "
    "percentage points (Mann-Whitney U, all pairwise p<10⁻⁴). The per-ethnicity fairness "
    "gap Δ_eth collapses from 12.29 pp (RF) and 11.34 pp (LR) to 5.78 pp (CNN) and 6.34 pp "
    "(DANN-Trust). CNN and DANN-Trust are statistically indistinguishable on overall accuracy "
    "(p=0.86) and on Δ_eth (p=0.26). CNN and DANN each significantly beat ANN on Δ_eth "
    "(p=0.001 and p=0.022 respectively). The overall AUC values (0.80–0.83 across all five "
    "models) sit above the source paper's published 0.77 for RF and 0.76 for LR [1, Table 10], "
    "for the same protocol reason as above (joint-stratified split versus LOSO). Separately, "
    "the five models' AUC values are tightly clustered relative to their differences in "
    "accuracy, suggesting that most of the accuracy advantage of the deep models comes from "
    "better-calibrated decision thresholds rather than from a fundamentally more separable "
    "representation."
)
replace_paragraph_text(d.paragraphs[149], NEW_B_NARR)

NEW_C_157 = (
    "The pattern is clearest on the South Asian subset, where Random Forest reaches only "
    "66.66% accuracy, 12.05 percentage points below the White group, replicating the source "
    "paper's finding that this group is hardest for its tree-based classifier [1, Table 8]. "
    "The deep models raise South Asian accuracy by roughly 8 to 10 points, closing most of "
    "that distance (ANN 75.07%, CNN 76.12%, DANN 74.99%)."
)
replace_paragraph_text(d.paragraphs[157], NEW_C_157)

NEW_C_158 = (
    "Our baseline gaps of 12.29 pp for Random Forest and 11.34 pp for Logistic Regression are "
    "roughly two to three times the 5 pp and 4 pp spreads in the published results "
    "[1, Table 8], most likely because each per-group accuracy is estimated here from between "
    "67 and 96 test utterances rather than from the full corpus under leave-one-speaker-out. "
    "Comparisons in this section are therefore between models evaluated under our protocol. "
    "The two baselines also disagree on which group is hardest for them individually: RF's "
    "weakest subset is South Asian (66.66%) while LR's is Black (64.71%). Because these are "
    "two different classifiers trained on the same features and the same split, this indicates "
    "that each baseline is finding a different demographic shortcut rather than one group "
    "being intrinsically harder to classify. Therefore, subset accuracy depends on the choice "
    "of model, not solely on the underlying acoustics."
)
replace_paragraph_text(d.paragraphs[158], NEW_C_158)

NEW_C_159 = (
    "Among the three deep models, CNN has the flattest per-ethnicity mean profile at 50 seeds, "
    "with only 1.33 percentage points separating its highest (White, 77.12%) and lowest "
    "(Black, 74.79%) group means, followed by DANN-Trust with 4.22 pp (White 78.46% vs Black "
    "74.24%). The 5.78 pp and 6.34 pp mean values reported for their Δ_eth in Table 2 are "
    "larger than these mean-of-means differences because Δ_eth is computed within each seed "
    "before averaging, and the identity of the extreme groups varies across seeds. The 50-seed "
    "protocol also stabilizes the per-group standard deviations: at 50 seeds every deep model "
    "sits between roughly 2 and 4 pp of per-group std, so the earlier appearance of one model "
    "being uniformly more stable per group than another does not hold up once seed count is "
    "increased."
)
replace_paragraph_text(d.paragraphs[159], NEW_C_159)

NEW_D_174 = (
    "Overall, Black speakers form the weakest AUC cell for three of the five models (RF 0.777, "
    "ANN 0.782, DANN 0.796). LR is an exception with its weakest cell on Black speakers "
    "(0.721, the lowest value of the table), consistent with its 64.71% accuracy for the same "
    "group in Table 3 and suggesting that a linear decision boundary is less proficient at "
    "separating Trustworthy from Neutral speech for this subset. Random Forest's weakest cell "
    "is South Asian (0.754), while its White cell is the strongest value in the table (0.889), "
    "giving it the widest AUC spread alongside the widest accuracy gap (Table 2). LR shows the "
    "reverse ordering, holding the widest AUC spread (0.152) but the narrower of the two "
    "baseline accuracy gaps, an early indication that the two measures do not rank models "
    "identically."
)
replace_paragraph_text(d.paragraphs[174], NEW_D_174)

NEW_D_176 = (
    "This reflects Section IV-C, where RF and LR disagree on which subset is harder to "
    "classify. RF's weakest group is South Asian in both tables, while LR's is Black in both. "
    "AUC does not depend on the decision threshold, so the difference is not a consequence of "
    "threshold placement. Each classifier appears to rely on a different demographic shortcut "
    "rather than one group being intrinsically harder to classify. The two models with the "
    "narrowest AUC spreads are also the two with the narrowest accuracy gaps: CNN at 0.055 "
    "and DANN-Trust at 0.065, against 0.079 for the ANN and 0.135 and 0.152 for the baselines. "
    "The same inflation seen in the accuracy gaps appears here: the source paper's RF AUC "
    "spans 6 pp across ethnicity under LOSO [1, Table 10] against 13.5 pp under our split, "
    "consistent with the smaller per-group samples described in Section IV-C."
)
replace_paragraph_text(d.paragraphs[176], NEW_D_176)

NEW_E_180 = (
    "A central methodological question of this paper is whether DANN-Trust is more stable "
    "across retraining than the CNN. Table 5 collects each model's seed-to-seed standard "
    "deviation on the two metrics that matter most when deciding fairness behavior: overall "
    "accuracy and the per-ethnicity fairness gap Δ_eth, now measured across 50 random seeds."
)
replace_paragraph_text(d.paragraphs[180], NEW_E_180)

NEW_E_183 = (
    "The 50-seed variances no longer support a stability advantage for DANN-Trust over the "
    "CNN. On accuracy, CNN std = 1.75% and DANN std = 1.71% (Levene's W-test p=0.97); on "
    "Δ_eth, CNN std = 2.73 pp and DANN std = 3.14 pp (p=0.33). LR's zero variance is a "
    "diagnostic on the harness rather than a finding about LR itself: a deterministic solver "
    "with a fixed random state produces identical solutions across reseeds. The apparent DANN "
    "stability advantage reported in an earlier five-seed version of this analysis "
    "(CNN Δ_eth std ≈ 2.87 pp vs DANN ≈ 0.89 pp) did not survive the seed-count increase; "
    "with only five seeds a single lucky run collapses the estimated std by a factor of three, "
    "and the effect disappears once each estimate is drawn from 50 runs. All three deep models "
    "do show significantly higher accuracy std than Random Forest (Levene's p<0.03 for each), "
    "but among the deep models themselves no pair differs significantly on the stability of any "
    "reported metric. The methodological lesson stands independent of the direction of the "
    "result: reporting fairness from a single training run, or from only a handful of seeds, "
    "can produce a stability claim that a larger seed count would overturn."
)
replace_paragraph_text(d.paragraphs[183], NEW_E_183)

NEW_FIG7 = (
    "Fig. 7 Fairness vs accuracy trade-off (mean ± std over 50 seeds). CNN and DANN-Trust "
    "occupy essentially the same region of the plane; their error bars overlap on both axes."
)
replace_paragraph_text(d.paragraphs[193], NEW_FIG7)

NEW_PROBE = (
    "To verify the adversarial mechanism directly, the frozen encoder output (32-dimensional "
    "in the no-BN configuration) is extracted on the test set and a new logistic-regression "
    "probe is trained to predict ethnicity from it. The probe reaches roughly 53% mean "
    "accuracy across seeds, against a chance level of 33.3% for three ethnicities. This "
    "indicates the encoder is not demographic-invariant in the strict sense, since a "
    "downstream classifier can still extract ethnicity signal from ϕ(x). Adversarial training "
    "does not fully erase ethnicity information from the encoder's output, but it does prevent "
    "the trust head from using that residual information to produce disproportionate per-group "
    "predictions, which is the fairness definition (consistent per-group accuracy) used "
    "throughout this paper. Narrowing this remaining leakage through richer adversarial heads, "
    "larger encoders, and pretrained speech representations is the implied direction for "
    "future work."
)
replace_paragraph_text(d.paragraphs[195], NEW_PROBE)

NEW_DISC_199 = (
    "At 50 seeds, all three deep models outperform the published Random Forest and Logistic "
    "Regression baselines on both accuracy and per-ethnicity fairness. Among the deep models, "
    "CNN and DANN-Trust are statistically indistinguishable on every headline metric "
    "considered here, so neither can be presented as strictly better than the other on this "
    "corpus."
)
replace_paragraph_text(d.paragraphs[199], NEW_DISC_199)

NEW_DISC_200 = (
    "Relative to Random Forest, ANN improved overall accuracy by 3.20 pp, CNN by 3.10 pp, "
    "and DANN-Trust by 3.16 pp (Mann-Whitney U, all p<10⁻⁴). Over the same 50 seeds, the "
    "per-ethnicity accuracy gap decreased from 12.29 pp under Random Forest to 5.78 pp for "
    "the CNN and 6.34 pp for DANN-Trust. The largest per-group improvement is on South Asian "
    "speakers, whose mean accuracy rises from 66.66% under Random Forest to 76.12% under CNN "
    "and 74.99% under DANN-Trust. These results support the interpretation that the ~71% "
    "accuracy reported in the original TIS study was not a ceiling imposed by the corpus but "
    "was partly attributable to the model classes originally evaluated. The improvement of the "
    "neural models is also consistent with previous research indicating that no single acoustic "
    "cue reliably characterizes vocal trustworthiness; the trait may depend on interactions "
    "among multiple features that nonlinear models capture more effectively than the published "
    "linear and tree-based baselines."
)
replace_paragraph_text(d.paragraphs[200], NEW_DISC_200)

NEW_DISC_201 = (
    "CNN and DANN-Trust could not be separated on point estimates of accuracy (76.15% vs "
    "76.21%, p=0.86) or on the mean per-ethnicity gap (5.78 pp vs 6.34 pp, p=0.26). At five "
    "seeds an earlier version of this analysis reported that DANN-Trust was substantially "
    "more stable than CNN on the fairness gap (std 0.89 pp vs 2.87 pp). At 50 seeds that "
    "advantage disappears: CNN std = 2.73 pp, DANN std = 3.14 pp, Levene's p=0.33 for the "
    "difference; the earlier low DANN estimate was a low-seed-count artefact rather than a "
    "property of adversarial training."
)
replace_paragraph_text(d.paragraphs[201], NEW_DISC_201)

NEW_DISC_202 = (
    "A separate finding at 50 seeds concerns the sex axis, which the corpus was not originally "
    "designed to control. Both CNN and DANN-Trust significantly reduce the female-vs-male "
    "accuracy gap relative to Random Forest (10.63 pp → 8.69 pp for CNN, p=0.003; → 8.48 pp "
    "for DANN-Trust, p=0.002); ANN does not (p=0.16). All three deep models simultaneously "
    "have significantly higher sex-gap variance than Random Forest (Levene's p<0.02), so any "
    "single training run of a deep model could over- or under-shoot the reported mean by "
    "several pp. The age gap is small for every model (2.9–4.8 pp) and only the ANN "
    "significantly beats Random Forest on it (p=0.046). No deep model is significantly "
    "different from any other on the age gap."
)
replace_paragraph_text(d.paragraphs[202], NEW_DISC_202)

NEW_DISC_203 = (
    "The ablation reported in Section IV-G explains one architectural choice in the "
    "DANN-Trust encoder. Retraining DANN with BatchNormalization inserted at the same "
    "positions as in the ANN encoder produces a significantly worse model on all three axes "
    "the ablation measures: overall accuracy (dann_no_bn 76.21% vs dann_full 74.30%, "
    "Mann-Whitney U p=0.002), mean Δ_eth (6.34 pp vs 8.06 pp, p=0.011), and Δ_eth variance "
    "(std 3.14 pp vs 3.86 pp, Levene's p=0.049). We interpret this as running batch statistics "
    "being unstable when the encoder's representation is being continuously reshaped by the "
    "adversarial signal on a small tabular dataset. This is the reason the DANN-Trust "
    "architecture reported in the main tables is the no-BN configuration, and it supports the "
    "broader point that architectural details such as normalization layers should be treated "
    "as first-order fairness variables, not incidental implementation choices."
)
replace_paragraph_text(d.paragraphs[203], NEW_DISC_203)

NEW_DISC_206 = (
    "The frozen-encoder probe qualifies but does not overturn the conclusion that CNN and "
    "DANN-Trust deliver comparable, and substantially better, ethnicity fairness than the "
    "baselines. Speaker ethnicity remained recoverable from the DANN encoder at ~53% accuracy, "
    "above the 33.3% chance rate. Adversarial training reduced, but did not eliminate, "
    "demographic information. Accordingly, DANN-Trust should not be described as "
    "demographically invariant or universally fair. Rather, on the TIS Corpus and under the "
    "accuracy-gap metric used here, both CNN and DANN-Trust produce among the smallest "
    "per-ethnicity disparities observed while maintaining competitive overall accuracy. The "
    "case for choosing one over the other on this corpus rests on considerations outside the "
    "measured metrics (implementation complexity, interpretability of the adversarial signal, "
    "and behavior on future related datasets)."
)
replace_paragraph_text(d.paragraphs[206], NEW_DISC_206)

NEW_LIMITS = (
    "Several limitations should guide the interpretation of these findings. First, the study "
    "still uses a single fixed train-test split. This design deliberately isolates variation "
    "caused by stochastic training but does not measure variation across different data "
    "partitions; future work should combine repeated seeds with repeated train-test splits. "
    "Second, the findings are limited to the TIS Corpus (1,152 utterances from three ethnic "
    "groups). Some demographic cells contain relatively few speakers, limiting the precision "
    "of per-group estimates. Third, the fairness metric Δ measures only the maximum-minus-"
    "minimum of per-group accuracy. It does not capture differences in error types; future "
    "studies should supplement accuracy gaps with per-group false-positive and false-negative "
    "rates and with equalized-odds or calibration-based metrics. Fourth, the ablation isolates "
    "BatchNormalization but does not systematically vary the gradient-reversal schedule, the "
    "adversarial-head capacity, or the encoder width; these remain plausible axes for further "
    "study. Fifth, results here concern speaker intent (produced trustworthiness) rather than "
    "listener perception; generalization to perception-labelled corpora is not established."
)
replace_paragraph_text(d.paragraphs[213], NEW_LIMITS)

NEW_CONC_215 = (
    "This project set out to reduce a demographic disparity in voice-trust classification and "
    "found that much of it was not a property of the data at all. The accuracy ceiling reported "
    "for this corpus belonged to the model class rather than the data, and most of the "
    "ethnicity disparity disappeared once that ceiling was lifted, before any fairness "
    "objective was applied. Under a 50-seed statistical protocol, CNN and DANN-Trust are "
    "indistinguishable on accuracy, ethnicity gap, and sex gap; a stability advantage for "
    "DANN-Trust that appeared at five seeds did not survive the seed-count increase. Both "
    "deep models additionally close a significant portion of the female-vs-male accuracy gap "
    "relative to the classical baselines. A supporting ablation shows that BatchNormalization "
    "in the DANN encoder significantly harms every fairness metric it was intended to help, "
    "so the reported DANN-Trust architecture omits it."
)
replace_paragraph_text(d.paragraphs[215], NEW_CONC_215)

NEW_CONC_216 = (
    "These findings carry three implications for speech-based AI. First, because the TIS "
    "Corpus labels speaker intent rather than listener perception, the observed performance "
    "differences across groups concern how speakers produce trustworthy intent, not how "
    "listeners judge their voices; systems trained on listener perceptions risk treating "
    "shared expectations about how a trustworthy voice should sound as objective properties of "
    "the speaker. Second, future systems should evaluate demographic performance not only by "
    "auditing average accuracy across groups but also by treating fairness metrics as "
    "distributions over independent retrains, with statistical tests on both means and "
    "variances. Third, architectural components that are usually considered incidental "
    "(normalization layers, adversarial-head width, gradient-reversal schedule) can dominate "
    "the fairness outcome and should therefore be reported and ablated alongside the loss "
    "function itself."
)
replace_paragraph_text(d.paragraphs[216], NEW_CONC_216)

# ---------- Insert Section IV-G (ablation) before V. DISCUSSION ----------
disc_idx = None
for i, p in enumerate(d.paragraphs):
    if p.text.strip().startswith("V. DISCUSSION"):
        disc_idx = i
        break

if disc_idx is not None:
    anchor = d.paragraphs[disc_idx - 1]
    # We insert paragraphs in reverse order after `anchor`
    # so the final visual order is (heading, table lines, narrative).
    reverse_lines = [
        (
            "Removing BatchNormalization from the DANN-Trust encoder is a triple-"
            "significant improvement: accuracy rises by 1.91 pp (Mann-Whitney U, "
            "p=0.002), the mean ethnicity gap tightens by 1.72 pp (p=0.011), and the "
            "seed-to-seed variance of that gap contracts as well (Levene's p=0.049). "
            "The interpretation adopted in Section III-F is that running batch "
            "statistics are unstable when the encoder representation is being "
            "continuously reshaped by adversarial pressure on a small tabular "
            "training set (~921 utterances). All main-body DANN-Trust results in "
            "Sections IV-B through IV-F use the dann_no_bn configuration.",
            None,
        ),
        ("p-value     |  0.002 (**)      |  0.011 (**)     |  0.049 (*)", None),
        ("dann_no_bn  | 76.21 ± 1.71 %   |  6.34           |  3.14", None),
        ("dann_full   | 74.30 ± 1.86 %   |  8.06           |  3.86", None),
        ("Variant     | Accuracy         | Δ_eth mean (pp) | Δ_eth std (pp)", None),
        (
            "Table 8: DANN encoder ablation. dann_full includes two BatchNormalization "
            "layers in the encoder; dann_no_bn omits both. 50 seeds each, same fixed "
            "80/20 split. p-values are two-sided Mann-Whitney U on means, and Levene's "
            "on variances. Bold entries are significant at p<0.05.",
            None,
        ),
        ("G. Ablation: BatchNormalization in the DANN encoder", "Subtitle"),
    ]
    for text, style in reverse_lines:
        insert_paragraph_after(anchor, text, style=style)

d.save(SRC)
print(f"Wrote updated document to: {SRC}")
