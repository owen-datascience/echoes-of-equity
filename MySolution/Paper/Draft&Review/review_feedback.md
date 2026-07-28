# Review Feedback — Research paper rough draft.

**Date:** 2026-07-28

---

## I. Introduction

### Grammar / clarity to fix line-by-line

- **1.1:** *"These rapid impressions of which are involuntarily can directly cause real consequential decision-making"* — ungrammatical. Suggest: *"These impressions are involuntary and can shape consequential decisions — who is hired, who gets the loan, who is believed."*
- **1.1:** *"all derive features from speech in which a growing number of decision-support tools attach consequences to them"* — the "in which... to them" is broken. Rewrite as in the markdown: *"...all derive features from the same acoustic signal that humans interpret socially."*
- **1.1:** *"social weights of that judgment"* — should be *"social weight"* (singular).
- **1.3:** *"Frame stability is thereby a required property for a deployer or regulatory"* — "regulatory" is an adjective. Should be *"deployer or regulator"*.
- **1.4:** The paragraph starting *"The Random Forest and Logistic Regression baselines..."* is a sentence fragment (no main verb). Add *"We compare five classifiers: ..."*.
- **1.4:** *"We address both problems of the Trustworthy intent in Speech Corpus"* — should be *"We address both problems **using** the ..."*.

### Content losses vs. the markdown

- The markdown's contributions bullet list gives concrete numbers (73.4/71.9%, 76.5±2.5%, 5.0pp gap, 0.89pp std). The PDF's 1.6 collapses these into a running paragraph, losing the punch. Restore the bulleted format with numbers — reviewers scan contribution lists first.
- The markdown opens with a stronger, tighter first sentence: *"The human voice is among the richest social signals we produce."* The PDF's *"Much of human life revolves around communication..."* is a generic hook. Use the markdown version.
- The markdown Introduction cites Ganin et al. (2016) properly for the DANN paradigm and includes the shortcut-amplification argument. The PDF is weaker on why non-linear models introduce a new bias risk — this motivation belongs before you introduce DANN.

### Structural

- 1.6 has *"Reproduction of the published baselines & A controlled comparison of five classifiers"* squashed into a single bullet. Split them.
- 1.7 Roadmap uses Roman numerals but the rest of the paper uses Arabic — fix once you pick a scheme.

---

## III. Dataset (labelled "4." in the draft)

### Structural

- The heading is *"4. Dataset and Features."* but the roadmap says Section III. Fix to Section III.
- **Table 1 has spilled across a page break** (page 10 → 11) and lost its formatting — the second half becomes free-flowing text: *"Black Younger 11 9 20 Black Older 5 3 8 South Asian Younger 10 10 20..."*. Wrap Table 1 in a `\begin{table}` (LaTeX) or "Keep with next" (Word) so it never breaks mid-body.

### Content losses vs. the markdown

- The PDF describes the acoustic feature set as *"fundamental frequency (mean and standard deviation of F0), voice-quality measures ... sentence duration, and long-term average spectrum descriptors (LTAS mean, standard deviation and slope)"*. This lists only ~10 features but then says all 60 columns are used. The markdown's **five families** breakdown (Pitch 10, Voice-quality 12, Formant 14, Spectral 8, LTAS 6) is much more informative and accounts for all 60. **Port the five-family paragraph in verbatim.**
- The markdown flags the older-cell imbalance visually via **Fig. 2** and describes stratification as the mitigation. The PDF mentions the imbalance but does not reference the figure.
- The PDF says older-adult recruitment used *"posters and word of mouth"*; the markdown says *"in-person word-of-mouth"*. Verify which matches the source paper and unify.

### Prose issues

- **4.1.1:** *"All experiments conducted are attributed to the Trustworthy Intent in Speech (TIS) Corpus"* — "attributed to" is wrong here. Use the markdown's *"All experiments are conducted on ..."*.
- **4.1.3:** The last sentence spills from page 12 to page 13 as *"Stratifying on intent alone, as the source paper effectively does under leave-one-speaker-out cross-validation, lets the test set's ethnicity composition drift..."* — long and hard to parse. Break into two sentences per the markdown.
- **4.2** *"Original analysis recap"* — good addition not in the markdown, but the LOSO / accuracy numbers here overlap with material also in Section V. Decide where it lives and remove the duplicate.
- **4.3** *"A note on label validity"* — this same caveat appears again in Discussion (VI-C limitation #2). Either keep it here as motivation and drop it from Discussion, or keep it only in Discussion. The current draft says it twice.

### Missing elements from the markdown

- No mention of `Speech_dataset_characteristics.csv` as the exact input file (the markdown names it).
- No mention of the LOSO caveat *"typically reduces all reported accuracies by approximately 1 to 3 percentage points."* — this is useful for reader calibration.

---

## VI. Discussion (labelled "9." in the draft)

### Blocking issues

- **Section 9.3 is truncated** — ends at *"Seeds:"* mid-sentence. Missing: the full 6-row Limitations table (Table 10 in the markdown), the three limitations discussed in prose (small N, self-rated labels, DANN leakage), and the closing paragraph about the sixth methodological limitation being the interesting research direction.
- **Missing "What Worked" subsection structure.** The markdown organises VI-A around four distinct findings (accuracy lift, fairness lift, reproducibility from adversarial training, acoustic findings replicate). The PDF's 9.1 collapses them into one long paragraph — reader loses the four-part scaffold. Restore bold sub-labels.

### Content losses vs. markdown

- The markdown VI-A explicitly notes *"Acoustic findings replicate"* (F0 mean, F0 SD, HNR, shimmer, CPP as dominant cues per Random Forest Gini importance). This independent replication of the source paper's acoustic finding is a *fourth* contribution to the Discussion and is missing from the PDF entirely.
- **Fig. 13 discussion is diluted.** The markdown VI-B gives the (accuracy, gap) plane a clear geometry: "lower-right is good", "green arrow points there", RF/LR in upper-left, ANN upper-right, CNN/DANN lower-right. The PDF has the numerical facts but not the visual story. Restore the geometry-first framing.
- The markdown notes that Fig. 13 *"marks the source paper's Random Forest LOSO result as a black star at (71%, 5.0pp)"* — key visual anchor for the headline claim. Missing from the PDF.

### Prose issues

- **9.1** opens *"All three neural models have outperformed both baselines"* — tense drift; use simple past: *"outperformed"* only. And *"our largest single beneficiary"* is awkward; markdown uses the cleaner *"The biggest beneficiary was the South Asian group"*.
- **9.1:** *"DANN-Trust posits the most consistent per-group profile"* — "posits" is the wrong verb. Use *"delivers"* or *"achieves"*.
- **9.2:** The distinction between **behavioural fairness (accuracy parity)** and **representational fairness (invariance)** is well made — keep this exactly as in the markdown (it is the strongest paragraph in the section). Cite Hardt et al. (2016) — the PDF does so; make sure the reference is in the bibliography.
- **9.2** spills onto page 35 with *"groups roughly equally in accuracy on this corpus. We cannot claim however that it no longer represents ethnicity internally."* — good sentence, but hyphenate "however" correctly: *"We cannot claim, however, that ..."*.

### Structural

- 9.3 will need Table 10 (the six-row limitations table from the markdown). Confirm the mitigations column matches what you actually plan to do — items 4 and 5 (λ sweep, k-fold CV) are the two cheapest to actually run before submission and would strengthen the paper if you did them, rather than list them as future work.

---

## Priority order for revision

1. Fix PDF export (word-per-line spacing) and delete the "Plan" / meeting-note front matter.
2. Complete the truncated 9.3 Limitations section (port Table 10 from the markdown).
3. Fix Table 1 formatting so it does not spill across the page break.
4. Reconcile section numbering (Roman vs. Arabic) across roadmap, headings, and cross-references.
5. Restore the four-part "What Worked" scaffold in the Discussion and the five-family feature breakdown in the Dataset.
6. Line-level grammar pass on the Introduction — several sentences do not parse.
7. Unify citation style (numeric IEEE-style is the target given the existing `[1]`, `[14]`, `[17]` in Dataset).

---

The markdown files under `MySolution/Paper/` are close to ready. The main task is bringing the PDF back in line with them, not rewriting from scratch.
