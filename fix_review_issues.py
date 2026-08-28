"""
One-shot: apply the 8 review fixes identified during content/results audit.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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

def find_paragraph_starting_with(d, prefix):
    for i, p in enumerate(d.paragraphs):
        if p.text.strip().startswith(prefix):
            return i, p
    return None, None

def replace_in_paragraph(paragraph, old_substr, new_substr):
    txt = paragraph.text
    if old_substr not in txt:
        return False
    new_txt = txt.replace(old_substr, new_substr)
    replace_paragraph_text(paragraph, new_txt)
    return True

d = Document(SRC)

# -------- Fix 7 (part A): Tables 2, 3, 5 std values --------
STD_MAIN = {
    "ANN":  "7.99 ± 3.56 pp",
    "CNN":  "5.78 ± 2.70 pp",
    "DANN": "6.34 ± 3.11 pp",
}
t = d.tables[1]
for i in range(1, len(t.rows)):
    model = t.rows[i].cells[0].text.strip()
    if model in STD_MAIN:
        set_cell_text(t.rows[i].cells[3], STD_MAIN[model])
        print(f"Table 2: {model} Δ column -> {STD_MAIN[model]}")

t = d.tables[2]
for i in range(1, len(t.rows)):
    model = t.rows[i].cells[0].text.strip()
    if model in STD_MAIN:
        set_cell_text(t.rows[i].cells[4], STD_MAIN[model])
        print(f"Table 3: {model} Δ column -> {STD_MAIN[model]}")

t = d.tables[4]
STD_MAP = {"ANN": "3.56", "CNN": "2.70", "DANN": "3.11"}
for i in range(1, len(t.rows)):
    model = t.rows[i].cells[0].text.strip()
    if model in STD_MAP:
        set_cell_text(t.rows[i].cells[2], STD_MAP[model])
        print(f"Table 5: {model} Δ_eth std -> {STD_MAP[model]}")

# -------- Fix 6: §IV-B rewording --------
_, p149 = find_paragraph_starting_with(
    d, "All three deep models exceed both baselines on overall accuracy"
)
if p149 is not None:
    replace_paragraph_text(p149, (
        "All three deep models exceed the RF baseline by 3.10 to 3.20 percentage "
        "points and the LR baseline by 4.29 to 4.39 pp (Mann-Whitney U, all pairwise "
        "p<10⁻⁴). The per-ethnicity fairness gap Δ_eth collapses from 12.29 pp (RF) "
        "and 11.34 pp (LR) to 5.78 pp (CNN) and 6.34 pp (DANN-Trust). CNN and "
        "DANN-Trust are statistically indistinguishable on overall accuracy (p=0.86) "
        "and on Δ_eth (p=0.26). CNN and DANN each significantly beat ANN on Δ_eth "
        "(p=0.001 and p=0.022 respectively). The overall AUC values (0.80–0.83 across "
        "all five models) sit above the source paper's published 0.77 for RF and 0.76 "
        "for LR [1, Table 10], for the same protocol reason as above (joint-stratified "
        "split versus LOSO). Separately, the five models' AUC values are tightly "
        "clustered relative to their differences in accuracy, suggesting that most of "
        "the accuracy advantage of the deep models comes from better-calibrated "
        "decision thresholds rather than from a fundamentally more separable "
        "representation."
    ))
    print("Fix 6: §IV-B baseline-range rewording applied")

# -------- Fix 2: §IV-C "1.33" -> "2.33" --------
_, p159 = find_paragraph_starting_with(
    d, "Among the three deep models, CNN has the flattest per-ethnicity"
)
if p159 is not None:
    ok = replace_in_paragraph(p159, "only 1.33 percentage points", "only 2.33 percentage points")
    print(f"Fix 2: 1.33->2.33 in §IV-C: {ok}")

# -------- Fix 1: §IV-D garbled logic --------
_, p174 = find_paragraph_starting_with(
    d, "Overall, Black speakers form the weakest AUC cell"
)
if p174 is not None:
    replace_paragraph_text(p174, (
        "Overall, Black speakers form the weakest AUC cell for four of the five "
        "models (LR 0.721, ANN 0.782, CNN 0.790, DANN 0.796), with LR's 0.721 the "
        "lowest value in the table. This is consistent with LR's 64.71% accuracy for "
        "the same group in Table 3 and suggests that a linear decision boundary is "
        "less proficient at separating Trustworthy from Neutral speech for this "
        "subset. Random Forest is the exception: its weakest cell is South Asian "
        "(0.754), while its White cell is the strongest value in the table (0.889), "
        "giving it the widest AUC spread alongside the widest accuracy gap (Table 2). "
        "LR shows the reverse ordering, holding the widest AUC spread (0.152) but the "
        "narrower of the two baseline accuracy gaps, an early indication that the two "
        "measures do not rank models identically."
    ))
    print("Fix 1: §IV-D weakest-AUC paragraph rewritten")

# -------- Fix 7 (part B): §IV-E std corrections --------
_, p183 = find_paragraph_starting_with(
    d, "The 50-seed variances no longer support"
)
if p183 is not None:
    ok = replace_in_paragraph(
        p183,
        "CNN std = 2.73 pp and DANN std = 3.14 pp",
        "CNN std = 2.70 pp and DANN std = 3.11 pp",
    )
    print(f"Fix 7B: §IV-E std correction: {ok}")

# -------- Fix 3: §IV-G Table 8 rows --------
_, p_full = find_paragraph_starting_with(d, "dann_full   |")
if p_full is not None:
    replace_paragraph_text(p_full, "dann_full   | 75.48 ± 2.15 %   |  8.68           |  4.40")
    print("Fix 3a: Table 8 dann_full row corrected")
_, p_nobn = find_paragraph_starting_with(d, "dann_no_bn  |")
if p_nobn is not None:
    replace_paragraph_text(p_nobn, "dann_no_bn  | 76.91 ± 1.84 %   |  6.55           |  3.14")
    print("Fix 3b: Table 8 dann_no_bn row corrected")

# -------- Fix 4: §IV-G narrative pp deltas --------
_, p_narr = find_paragraph_starting_with(
    d, "Removing BatchNormalization from the DANN-Trust encoder"
)
if p_narr is not None:
    ok1 = replace_in_paragraph(p_narr, "accuracy rises by 1.91 pp", "accuracy rises by 1.43 pp")
    ok2 = replace_in_paragraph(p_narr, "the mean ethnicity gap tightens by 1.72 pp", "the mean ethnicity gap tightens by 2.13 pp")
    print(f"Fix 4: §IV-G narrative pp deltas: {ok1}/{ok2}")

# -------- Fix 5: §V-A discussion echo of Table 8 --------
_, p_disc_ablation = find_paragraph_starting_with(
    d, "The ablation reported in Section IV-G explains one architectural choice"
)
if p_disc_ablation is not None:
    ok1 = replace_in_paragraph(
        p_disc_ablation,
        "overall accuracy (dann_no_bn 76.21% vs dann_full 74.30%",
        "overall accuracy (dann_no_bn 76.91% vs dann_full 75.48%",
    )
    ok2 = replace_in_paragraph(
        p_disc_ablation,
        "mean Δ_eth (6.34 pp vs 8.06 pp",
        "mean Δ_eth (6.55 pp vs 8.68 pp",
    )
    ok3 = replace_in_paragraph(
        p_disc_ablation,
        "Δ_eth variance (std 3.14 pp vs 3.86 pp",
        "Δ_eth variance (std 3.14 pp vs 4.40 pp",
    )
    print(f"Fix 5: §V-A ablation echo: {ok1}/{ok2}/{ok3}")

# -------- Fix 7 (part C): §V-A discussion CNN/DANN std mention --------
_, p_disc_stability = find_paragraph_starting_with(
    d, "CNN and DANN-Trust could not be separated"
)
if p_disc_stability is not None:
    ok = replace_in_paragraph(
        p_disc_stability,
        "CNN std = 2.73 pp, DANN std = 3.14 pp",
        "CNN std = 2.70 pp, DANN std = 3.11 pp",
    )
    print(f"Fix 7C: §V-A std correction: {ok}")

# -------- Fix 8: §V-B header typo --------
_, p_vb_header = find_paragraph_starting_with(d, "B. What does")
if p_vb_header is not None:
    ok1 = replace_in_paragraph(p_vb_header, "'Fair' means here", "'Fair' mean here")
    ok2 = replace_in_paragraph(p_vb_header, "“Fair” means here", "“Fair” mean here")
    print(f"Fix 8: §V-B header typo: {ok1 or ok2}")

d.save(SRC)
print(f"\nSaved: {SRC}")
