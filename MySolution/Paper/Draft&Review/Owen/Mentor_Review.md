# Mentor Review — Echoes of Equity (Rough Draft)

**Date:** 2026-08-04

---

## Overall Impression (read this first)

You have picked a strong, timely topic (fairness in voice AI), you have real experimental results, and your writing shows that you understand what you did and why it matters. The **stability finding** (DANN-Trust has ~1/3 the seed-to-seed variance of the CNN at the same average fairness gap) is a genuinely interesting contribution — I have not seen many high-school papers frame fairness as a *variance* problem, and you explain it well.

The current PDF, however, reads like a draft that has been printed **before you finished assembling it**. Several sections that exist in your markdown files did not make it into the PDF, several sections still contain literal "TODO" notes, and the section numbering breaks in a few places. **None of this is hard to fix**, but a judge who only reads the PDF will notice all of it immediately. Fix the "assembly" problems first (Section A below), then work through the "content" problems (Section B), then the polish items (Section C).

---

## A. Assembly problems (fix these BEFORE anything else)

These are things where the PDF does not match what you actually wrote or intended. A judge reading only the PDF will think entire sections are missing or unfinished.

### A1. The Related Work section is empty in the PDF
- On PDF page 7 there is only the heading "2. Related Works" and the sub-heading "2.1 Acoustic correlates of vocal trustworthiness / social voice perception." **There is no body text.**
- But `MySolution/Paper/section_II_related_work.md` contains a full, well-written Related Work section with three subsections (II-A voice acoustics, II-B demographic bias, II-C adversarial debiasing).
- **Action:** Paste the content of `section_II_related_work.md` into the PDF. This section is required by every competition rubric — without it the paper looks incomplete.

### A2. The cover page, title, author line, and abstract are missing from the PDF
- The PDF starts with a page that just says "Introduction" and then jumps into §1.1.
- Your `section_0_front_matter.md` has a full cover page, title, author block, abstract (263 words), and index terms — none of it is in the PDF.
- **Action:** Paste the front-matter content in front of §1. Fill in the `[FILL IN]` placeholders (name, school, advisor, email) before printing.

### A3. Multiple "TODO" placeholders are still visible in the paper
- Sections 5.4, 5.5, 5.6, and 5.7 in the PDF are literally blocks of "TODO (data collection): ..." text that describe experiments you have not yet run (per-age-group, per-sex, confusion matrices, feature importance, ablation).
- A submitted paper must never contain the word "TODO" or say "The change is approximately 20 lines."
- **Action:** Either (a) run those experiments and put the real numbers in, or (b) delete the empty sections and move a brief note ("per-age and per-sex breakdowns are left to future work") into the Limitations section. Option (a) is better if you have time — the code changes are small and the results will strengthen the paper.

### A4. Section numbering is broken in three places
1. The **intro paragraph of Section 5** says *"Section 6.1 describes the shared experimental setup … Section 6.2 presents overall classification performance…"* — every "6.x" here should be "5.x". You are inside Section 5, not Section 6.
2. **Sections jump from 5.3 straight to 5.8** (5.4, 5.5, 5.6, 5.7 are the TODO stubs). After you fix A3, renumber so the sections run 5.1, 5.2, 5.3, 5.4, … with no gaps.
3. The Discussion has a section labelled **"9.4 Ethical considerations"** — this should be "6.4" because you are inside Section 6.

### A5. Citation style is mixed (author-year AND numeric)
- Some places use IEEE-style numbers: `[1]`, `[22]`, `[23]`, `[26]`, `[37]`.
- Other places use author-year: "Maltezou-Papastylianou, Scherer & Paulmann, 2025a", "Ganin et al. (2016)", "Hardt et al., 2016".
- Your own `references.md` has a "Citation map" table at the bottom that tells you exactly which author-year cite to replace with which `[N]`. **Use it.** Pick one style (numeric `[N]` is standard for IEEE competitions) and do a find-and-replace across the whole paper.
- Also: I see `[37]` cited in §1.3 ("Underspecification … [37]") but your references.md only lists `[1]`–`[35]`. Either add the missing reference or fix the number.

### A6. Wrong section reference in the Discussion
- Discussion §6.2 says *"The frozen-encoder probe of Section 5.5…"* — the probe is actually described in **Section 5.9** in your current draft. Fix the cross-reference (or, better, add a stable label like "the probe of §5.9" once and update it in one place if you renumber).

---

## B. Content problems (fix these before final submission)

These are places where something in the PDF is factually off, contradicts your own data, or leaves a claim un-supported.

### B1. Table 4 (per-ethnicity AUC) does not match `results.json`

The paper's Table 4 shows the CNN as almost perfectly uniform: **White 0.818, Black 0.818, South Asian 0.808**, and Section 5.8 leans on that to say CNN is "the most internally consistent model across ethnicities."

But `MySolution/Analysis/results.json` gives, for CNN:
- White AUC = 0.8308
- Black AUC = **0.7846**
- South Asian AUC = 0.8323

So the CNN's Black AUC is actually its *lowest* cell, not its highest — the opposite of what the paper says. Similar smaller mismatches exist for RF, ANN, and DANN.

- **Action:** Regenerate Table 4 directly from `results.json` (the same file every other table came from). Then rewrite the paragraph under Table 4 to describe what the *real* numbers show. If CNN's per-ethnicity AUC is *not* uniform, that is still fine to say — you just need the story to match the data.

### B2. One claim in `section_V_experiments.md` contradicts the PDF (and the data)

The markdown file `section_V_experiments.md` line 51 says DANN-Trust has *"only 0.40pp between the best (Black, 73.82%) and worst (White, 75.62%)"* group means. The PDF version correctly says *"only 1.80 percentage points."* The correct number is 75.62 − 73.82 = **1.80pp**.

- **Action:** Update the markdown file so the two sources agree. Anyone who reads both will spot the contradiction.

### B3. The GRL ablation is promised but not run
Your paper's core claim about DANN-Trust is that the **gradient-reversal layer** is what causes the fairness-gap variance to shrink. Section 5.7 (the "TODO — ablation" block) plans to test this by running DANN with the GRL replaced by an identity function (i.e., pure multi-task learning). Until that ablation is run, you cannot rule out the possibility that just adding an ethnicity-prediction head — with no gradient reversal at all — would have given the same variance reduction.

- **Action:** Run the ablation. Your `dann_trust_model.py` is already set up for this — just replace `GradientReversalLayer(...)` with `layers.Lambda(lambda x: x)` and rerun with the 5 seeds. This is one of the highest-value experiments you have left, because it directly tests the paper's headline mechanism.

### B4. The "3-layer ANN was chosen because deeper didn't help" claim has no evidence
Section 4.4 says: *"the model was deliberately kept at three layers as previous experiments showed that additional layers led to diminishing results and overfitting."* If those previous experiments exist, put a small table in the appendix (or one sentence with numbers). If they don't, soften the wording: *"we did not go deeper because the training set of 921 samples would over-fit a larger network."*

### B5. Section 3.1.2 has a duplicated paragraph
The paragraph beginning *"The released feature set covers fundamental frequency (mean and standard deviation of F0)…"* mostly repeats the paragraph two above it (*"The 60 acoustic columns fall into six families…"*). Delete one of them — you're saying the same thing twice.

### B6. The intro promises 5 contributions but numbering isn't consistent with the abstract
The Abstract highlights three things (reproduction, deep-model lift, DANN-Trust variance reduction). The Introduction §1.6 lists **five** contributions. That's fine — the intro can expand — but the third bullet ("Reproduction of the published baselines & A controlled comparison of five classifiers") is actually two contributions crammed into one bullet, which makes the count feel off. Consider splitting it, or dropping the "&" so it reads as one clean item.

### B7. Small factual thing about the encoder-probe result
You report the probe accuracy as 53.7%, then say *"roughly 20 percentage points of leakage above chance"* (chance = 33.3%, so 53.7 − 33.3 = 20.4pp — correct). Then in §6.2 you say the same figure is "approximately 54%". Pick one number (53.7% is more precise) and use it consistently. Also, the probe is **one specific run** — if you have time, average it across the 5 seeds too, so this number gets an uncertainty bar like everything else in the paper.

---

## C. Polish (fix these once B is done)

### C1. Grammar and phrasing
A few sentences in the Introduction are hard to parse on first read:
- §1.1: *"These rapid impressions of which are involuntarily can directly cause real consequential decision-making"* — the "of which are involuntarily" is broken. Suggested rewrite: *"These rapid, involuntary impressions can directly influence consequential decisions:"*
- §1.1: *"Any system that takes voice as an input could thereby be subject to the same signal that human listeners interpret socially."* — clearer as *"Any system that uses voice as input inherits the same social signal that human listeners react to."*
- §1.3: *"Underspecification, the condition in which many models perform near-identically well on held-out evaluation, but exhibit unexpected results in real-world domains"* — fine, but "unexpected results" is vague; say "very different behaviour" or "divergent generalisation."
- §3.3: *"and a model that separates the two conditions well need not track the impression a listener would form"* — good sentence, keep it.

Consider running the paper through a grammar checker (Grammarly or Word's built-in). About 20–30 small comma/word-choice fixes are needed but none are fatal.

### C2. Repetition of the "5pp gap" framing
The 5pp per-ethnicity gap is mentioned in the abstract, §1.3, §1.4, §1.5, §3.2, §4.3, §5.2, §5.3, §6.1, and §7. That's fine for the abstract, intro, and conclusion, but you can shorten some of the middle mentions — right now the same sentence appears three or four times.

### C3. Figure captions should stand alone
Judges often skim figures first. Make sure every caption starts with a sentence that describes what the figure shows (Fig 3 does this well; Fig 9 and Fig 13 could be tightened). See `figures_to_draw.md` — your own production checklist already says this.

### C4. Consistency in model naming
The paper uses "DANN-Trust" (with a hyphen) most of the time but "DANN" in some tables and figure legends. Pick one (DANN-Trust is more descriptive) and use it everywhere, including in Table 3, Table 5, Table 9, and the figures.

### C5. Front-matter placeholders
When you paste in `section_0_front_matter.md`, remember to replace `[FILL IN — Student Name]`, `[FILL IN — High School, City, State, Country]`, and `E-mail: [FILL IN]` with your real information. It is easy to submit with placeholders still in place — check twice.

### C6. The 3 duplicated pages in your PDF
The PDF has full blank pages that just contain a big heading like "Introduction", "Related Work", "Dataset", "Methodology", "Experiment", "Experiment TODO's.", "Discussion". These look like chapter dividers from your writing tool. In a competition paper they should be removed — sections should begin directly with their numbered heading (e.g., "1. Introduction") on the same page as the body text.

---

## D. What you have already done well (keep it)

I want to be clear these things are strong and should not be changed:

1. **The stability framing is genuinely novel.** Most fairness papers report one number; you report five and quantify their spread. That is the right instinct.
2. **Honest reporting of the frozen-encoder probe result.** Many students would have hidden a 53.7% probe accuracy or spun it as "encoder is invariant." You correctly say it is *not* invariant and explain what the model does achieve instead. This is exactly the kind of scientific honesty a judge rewards.
3. **The RF vs LR "different demographic shortcut" observation in §5.3** is a genuinely useful insight that goes beyond what the source paper reported. Keep that paragraph.
4. **The reproduction of the source paper baselines within ±2pp** shows you can replicate other people's work, which is a mark of a careful researcher.
5. **Joint stratification on (intent × ethnicity)** and the explicit explanation of *why* you did it (to stop per-ethnicity accuracy variance from ballooning) — this is a graduate-level methodology choice, well explained.
6. **The Limitations section is honest and thorough.** You list five real limitations and don't oversell the results. Keep this.
7. **The Ethical Considerations section** (once you fix the "9.4 → 6.4" numbering) makes an important point about physiognomy-style automated judgement. Judges care about ethics discussion in fairness papers.

---

## E. Priority checklist for your next work session

If you only have a few hours, do these in order:

1. ☐ Paste in the missing Related Work section (fixes A1)
2. ☐ Paste in the front matter and delete blank divider pages (fixes A2, C6)
3. ☐ Delete or fill in every "TODO" block in Section 5 (fixes A3)
4. ☐ Fix "6.x → 5.x" numbering in the §5 intro and "9.4 → 6.4" in the Discussion (fixes A4)
5. ☐ Run the citation find-and-replace using your own `references.md` map (fixes A5)
6. ☐ Regenerate Table 4 (AUC) from `results.json` and update the paragraph beneath it (fixes B1)
7. ☐ Run the GRL ablation and add its row to Section 5.7 (fixes B3)
8. ☐ Delete the duplicated paragraph in §3.1.2 (fixes B5)
9. ☐ Grammar pass on the Introduction (fixes C1)
10. ☐ Fill in your name, school, advisor, e-mail in the front matter (fixes C5)

If items 1–5 are done, the paper is *submittable*. If 6–10 are also done, it is *competitive*.

---

## F. A note on the writing quality

Writing at this level (introducing a proper problem, motivating it with real-world examples, citing 30+ sources, replicating a baseline, adding three new architectures, running a multi-seed protocol, honestly reporting a partial-failure probe result) is well above what most high-school papers attempt. The core science here is solid. The reason this review is long is that the *presentation* is not yet doing your *science* justice — once the assembly and citation problems are fixed, the paper should read as a serious, professional-quality piece of work.

Good luck with the submission.
