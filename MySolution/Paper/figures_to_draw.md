# Architecture diagrams — drawing spec (Figs 1, 4, 5, 6)

These four figures are referenced in §I and §IV of the paper but cannot be auto-generated from data. This document specifies each one precisely enough that you (or a designer) can produce it in **drawio** (recommended for speed), **TikZ** (recommended if the paper goes to LaTeX), or **Inkscape / Figma** (recommended for visual polish) without re-deriving any structural decision.

Each section below has: (a) caption, (b) components and their exact dimensions, (c) layout and reading order, (d) annotations / labels to place, (e) ASCII sketch showing rough relative positions, (f) where the figure appears in the paper.

Recommended drawio output settings: **PNG @ 200 DPI, transparent background, 1600x900 px max** for half-page figures and **1600x500 px** for banner figures (Fig. 1).

---

## Fig. 1 — Overall pipeline (banner figure for §I)

**Caption draft.** *"Pipeline of the *Echoes of Equity* study. The publicly released TIS Corpus (1,152 utterances, 96 speakers across three ethnic backgrounds, two age groups, both sexes) is processed through VoiceLab to a 60-dimensional acoustic feature vector per utterance. Five classifiers — Random Forest, Logistic Regression, Artificial Neural Network, 1D-CNN and our Domain-Adversarial DANN-Trust — are evaluated on the same joint-stratified 80/20 split with multi-seed averaging. The output is a per-demographic accuracy + fairness-gap report that feeds the comparison in Section V."*

**Components, left to right.**
1. **Dataset block (left).** A small cylinder/disk icon labelled "TIS Corpus — 1,152 WAV". Three rows below it: "White (40 speakers)", "Black (28 speakers)", "South Asian (28 speakers)".
2. **Feature-extraction block.** A rectangle labelled "VoiceLab (Praat backend)" with arrow into a small matrix icon labelled "60 acoustic features x 1,152 utterances".
3. **Split block.** A rounded rectangle labelled "Joint-stratified 80/20 split (split seed = 42)" with two arrows out: 921 train (thicker arrow) and 231 test.
4. **Model column (vertical stack of 5 boxes, centre-right).** Five rectangles stacked top-to-bottom in this order: RF (blue), LR (orange), ANN (green), CNN (red), DANN-Trust (purple). The DANN box should be visually emphasised — bolder border, or a slight glow/halo — because it is the paper's contribution.
5. **Evaluation block (right).** A rectangle labelled "Per-demographic eval (5 seeds)" outputting two arrows: one to "Accuracy / AUC" and one to "Fairness gap Δ".
6. **Report block (far right).** A document icon labelled "results.json + Figs 9, 13 + Tables 3–10".

**Arrows.** All arrows are unidirectional left-to-right. Use grey, thickness ~2px. Where five model boxes feed into one evaluation block, draw five separate arrows merging at the evaluation block (do *not* use a single bundled arrow — readers should see that all five models run in parallel on the same data).

**Annotations.**
- Below the split block, in small text: `seed=42 (split), seeds=42-46 (model init)`.
- Above the DANN box, in small bold text: `headline contribution`.
- A small thought-bubble or note over the evaluation block: `also: probe ethnicity from DANN encoder -> Section V-I`.

**ASCII sketch.**
```
+--------+    +------------+    +--------------+         +------+
|  TIS   |--->|  VoiceLab  |--->|  Joint-strat |--921--->| RF   |
| Corpus |    |   60 feats |    |   80/20      |         +------+
| 1,152  |    +------------+    |   split42    |--231--->| LR   |   +----------+
| WAV    |                      +--------------+         +------+   | Per-demo |
+--------+                                               | ANN  |-->|  eval    |--> results.json
                                                         +------+   | (5 seeds)|        +
                                                         | CNN  |   |          |   Figs 9, 13
                                                         +======+   |          |   Tables 3-10
                                                         | DANN |   | probe->VI|
                                                         +======+   +----------+
                                                         headline
```

**Appears in.** §I, after Paragraph 4 (referenced as "Fig. 1 summarises this pipeline").

---

## Fig. 4 — ANN architecture (referenced in §IV-C)

**Caption draft.** *"Architecture of the ANN trust classifier. Input is the 60-dimensional acoustic feature vector; three hidden layers (Dense 128, 64, 32) with ReLU activations, batch normalisation and dropout produce a compressed representation that is passed through a sigmoid output for the trustworthy-intent probability."*

**Components, left to right.**
1. **Input layer.** Tall narrow rectangle labelled `x in R^60`. Below it: "60 acoustic features".
2. **Hidden block 1.** Rectangle labelled `Dense(128) + ReLU`. Below it: `BatchNorm`. Below that: `Dropout(0.3)`. Three stacked sub-rectangles or a single labelled box with three lines inside.
3. **Hidden block 2.** `Dense(64) + ReLU` + `BatchNorm` + `Dropout(0.3)`.
4. **Hidden block 3 (bottleneck).** `Dense(32) + ReLU`. Mark this with a slightly different colour or border — it is the compressed representation that DANN-Trust later wraps an adversary around.
5. **Output layer.** Small rectangle labelled `Dense(1) + Sigmoid`.
6. **Output indicator.** Arrow to a probability dial or `y_hat in [0, 1]` label, with annotation `Neutral <- 0 | 1 -> Trustworthy`.

**Annotations.**
- Below each Dense block, write the parameter count: `(60x128)+128 = 7,808` for h1; `(128x64)+64 = 8,256` for h2; `(64x32)+32 = 2,080` for h3; `(32x1)+1 = 33` for output. Total: **18,177 parameters** — write this as a small "Total params: 18.2k" label at the bottom.
- Above the bottleneck block, small italic label: *"encoder bottleneck — re-used in DANN-Trust (Fig. 6)"*.

**ASCII sketch.**
```
   x in R^60        h1                  h2                 h3 (bottleneck)       y_hat
                                                          ============
+---------+   +------------+    +------------+    +------|| Dense32 ||----+   +----------+
| 60      |-->| Dense 128  |--->| Dense 64   |--->|      ============     |-->| sigmoid  |--> y_hat
| feats   |   | ReLU+BN+Dr |    | ReLU+BN+Dr |    |        ReLU           |   |          |
+---------+   +------------+    +------------+    +-----------------------+   +----------+
                 (7.8k)            (8.3k)              (2.1k)                  (33)
                                                        ^
                                            re-used in DANN encoder
              Total: 18.2k params,  Adam lr=1e-3, batch=32, 50 epochs
```

**Appears in.** §IV-C, after the equation block.

---

## Fig. 5 — 1D-CNN architecture (referenced in §IV-D)

**Caption draft.** *"Architecture of the 1D-CNN trust classifier. The 60-feature acoustic vector is reshaped to a single-channel length-60 'signal' and processed by two stacked 1D convolutions (64 filters of width 3, then 32 filters of width 3) before a Dense 64 head and sigmoid output."*

**Components, left to right.**
1. **Input layer.** A rectangle labelled `X in R^{60 x 1}` (input acoustic vector reshaped as length-60 single-channel signal).
2. **Conv block 1.** Show a sliding kernel diagram: a width-3 grey rectangle overlaying part of the input row, with an arrow into a stack of 64 output feature maps (small horizontal bars labelled `(58, 64)`). Below: `Conv1D(64, k=3) + ReLU + BN + Dropout(0.2)`.
3. **Conv block 2.** Width-3 kernel sliding over the previous output, into a stack of 32 feature maps labelled `(56, 32)`. Below: `Conv1D(32, k=3) + ReLU`.
4. **Flatten layer.** Visual: 56 x 32 grid collapsed into a single horizontal bar labelled `1,792` units.
5. **Dense head.** `Dense(64) + ReLU` block.
6. **Output.** `Dense(1) + Sigmoid` and the `y_hat in [0, 1]` output arrow.

**Annotations.**
- Above Conv block 1, small italic note: *"kernel slides across **arbitrarily ordered** features — see Section IV-D for caveat"*. This is important because the paper explicitly justifies the CNN as a depth-control, not a locality-bias model.
- Below the flatten output, parameter count: `(56x32)+1 = 1,792 inputs into Dense 64 = 114,816 + 64 = 114,880 params at the Dense layer`. Total model: **~119k params**, mostly in the Dense head.

**ASCII sketch.**
```
   X in R^{60x1}     C1                       C2              Flatten        Dense        y_hat
                   +-------------+         +-------------+
+----------+    [#]| Conv1D 64x3 |->(58,64)| Conv1D 32x3 |->(56,32)-+
|   60     |--->[#]| ReLU+BN+Drop|         | ReLU        |          |      +---------+    +---+
| features |    [#]+-------------+         +-------------+          +----->| Dense 64|--->| s |--> y_hat
+----------+     kernel slides over                                        | ReLU    |    +---+
                 **arbitrary** feature                              (1,792)+---------+
                 triples - depth, not locality                              (114.9k)
              Total: ~119k params, Adam lr=1e-3, batch=32, 50 epochs
```

**Appears in.** §IV-D, after the equation block.

---

## Fig. 6 — DANN-Trust architecture (HERO figure for §IV-E)

**Caption draft.** *"Architecture of DANN-Trust, the domain-adversarial network. A shared encoder G_f produces a 32-dimensional representation phi(x) that feeds two heads: a trust head G_y predicting trustworthy intent, and a domain head G_d predicting speaker ethnicity. The Gradient Reversal Layer (GRL) between the encoder and the domain head acts as the identity on the forward pass and multiplies the upstream gradient by -lambda on the backward pass, forcing the encoder to make phi(x) less informative about ethnicity. Lambda is ramped from 0 to 1 across training using Ganin et al.'s sigmoid schedule."*

> **This is the paper's visual centrepiece — give it the most space and the most polish of any figure.** Ideally takes a full column width in a 2-column conference paper, or half-page in a single-column format.

**Components.**

Three colour-coded branches sharing a common root (the encoder):

1. **Encoder G_f (blue, central column).**
   - Input rectangle: `x in R^60`
   - Box: `Dense(128) + ReLU + BN + Dropout(0.3)`
   - Box: `Dense(32) + ReLU + BN`
   - Output node labelled `phi(x) in R^32` (this is the bottleneck — make it visually prominent, e.g., a circular or diamond node)

2. **Trust head G_y (green, top branch).** From phi(x):
   - Box: `Dense(16) + ReLU + Dropout(0.2)`
   - Box: `Dense(1) + Sigmoid`
   - Output: `y_hat in [0, 1]` labelled "Trustworthy probability"
   - **Gradient arrow** (green, dashed, flowing backward from y_hat through the head to phi): label `dL_trust / dphi` — the encoder receives a **normal** trust gradient.

3. **Domain head G_d (red/orange, bottom branch).** From phi(x):
   - **GRL block** (special — make it stand out, e.g., red border or hatched fill): label `Gradient Reversal Layer`, with small note `forward: x -> x; backward: grad -> -lambda*grad`.
   - Box: `Dense(16) + ReLU + Dropout(0.2)`
   - Box: `Dense(3) + Softmax`
   - Output: `z_hat in {White, Black, South Asian}` labelled "Ethnicity probability"
   - **Gradient arrow** (red, dashed, flowing backward from z_hat through the GRL to phi): label `-lambda * dL_dom / dphi` — note the minus sign is the GRL's effect.

4. **Bottom legend strip.** Show the loss equation:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{trust}}(\hat y, y) \;-\; \lambda \cdot \mathcal{L}_{\text{dom}}(\hat z, z)$$
   And the lambda schedule:
   $$\lambda_p = \lambda_{\max} \cdot \left( \tfrac{2}{1 + e^{-10p}} - 1 \right), \quad p = \tfrac{\text{epoch}}{80}$$

**Annotations.**
- Inside the GRL block: *"identity on forward, x(-lambda) on backward"*.
- Above the trust head: *"learn to predict y"*.
- Above the domain head: *"learn to predict z; encoder pushes against this"*.
- A small green arrow showing forward flow and a small red arrow showing backward flow, labelled in the legend.

**ASCII sketch.**
```
                                                  ===========================
                                                  || Trust head G_y (green)||
                                                  ||                       ||
                                              +-->|| Dense16 -> Dense1+s   ||--> y_hat
                                              |   ||   L_trust(y_hat, y)   ||
                                              |   ===========================
                                              |
   x in R^60                                  |
  +------+    +----------+    +---------+   *  phi(x)
  |      |--->| Dense128 |--->| Dense32 |---+  in R^32      ==================================
  | 60   |    | ReLU+BN  |    | ReLU+BN |   |               || Domain head G_d (red)         ||
  |feats |    | +Drop0.3 |    |         |   |               ||  +-----------------+           ||
  +------+    +----------+    +---------+   +-------------->||  | GRL: fwd id,    |->Dense16->||
              \---- Encoder G_f (blue) ----/                ||  | bwd x(-lambda)  |  Dense3+sm||
                                                            ||  +-----------------+        |  ||
                                                            ||                          z_hat in 3
                                                            ||              L_dom(z_hat, z) ||
                                                            ==================================

   Total loss:   L = L_trust(y_hat, y)  -  lambda * L_dom(z_hat, z)
   Schedule:     lambda(p) = lambda_max * (2 / (1 + exp(-10*p)) - 1),   p = epoch / 80
   lambda_max = 1.0, Adam lr = 1e-3, batch = 32, 80 epochs
```

**Appears in.** §IV-E, after the architecture description (between Steps 12 and 14 in the plan's numbering — i.e., right before the loss-function equation).

---

## Production checklist (run before submitting figures)

- [ ] All four figures rendered at >=200 DPI as PNG, *also* exported as SVG/PDF for LaTeX-friendly vector inclusion.
- [ ] Captions appear *below* each figure in the final paper (IEEE convention).
- [ ] Caption first sentence states what the figure shows; subsequent sentences add commentary. Captions are self-contained — a reader who reads only the figures and captions should follow the paper's argument.
- [ ] Colour-blind safe palette: prefer the same palette as the data-driven figures (`seaborn colorblind`, a.k.a. Tol's vibrant 10). For Fig. 6 specifically: blue encoder / green trust / red domain. Avoid pure red+green confusion by adding texture or shape variation if printing in greyscale.
- [ ] All text inside figures is sans-serif and at least 9pt at final reproduction size.
- [ ] Arrows are at least 1.5pt wide. Forward-flow arrows solid; gradient-flow arrows dashed.
- [ ] The DANN-Trust GRL block in Fig. 6 is visually distinguished (border, fill, or icon) — it is the paper's single most important architectural element.
- [ ] All filenames follow the `figXX_short_name.png` convention used by the data-driven figures (e.g., `fig01_pipeline.png`, `fig04_ann_arch.png`, `fig05_cnn_arch.png`, `fig06_dann_arch.png`) and are saved into `Paper/figures/`.
