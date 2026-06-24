# IV. Methodology

This section describes the five classifiers we compare on the TIS Corpus. Section IV-A formalises the trustworthy-intent classification problem and the fairness metric we optimise alongside accuracy. Section IV-B reproduces the linear and tree-based baselines from Maltezou-Papastylianou et al. [1]. Sections IV-C and IV-D introduce two deep models, an Artificial Neural Network (ANN) and a one-dimensional Convolutional Neural Network (1D-CNN), that lift accuracy above the published 71% baseline while collapsing the per-ethnicity accuracy gap. Section IV-E introduces our headline contribution, **DANN-Trust**, a domain-adversarial network whose gradient-reversal layer pushes the shared encoder toward ethnicity-invariant features and dramatically reduces the variance of fairness behaviour across random initialisations. Section IV-F lists every evaluation metric.

## IV-A Problem Formulation

Let $x \in \mathbb{R}^{d}$ denote the acoustic feature vector of one utterance, where $d = 60$ covers the pitch, voice-quality, formant, spectral and long-term-spectrum families extracted with VoiceLab [15] (Section III-B). Let $y \in \{0, 1\}$ be the binary trustworthy-intent label (0 = Neutral, 1 = Trustworthy) and let $z \in \{1, \ldots, K\}$ be a discrete demographic label, with $K = 3$ for ethnicity (White, Black, South Asian).

We seek a classifier $f_{\theta} : \mathbb{R}^{d} \to [0, 1]$ that maximises $P(y \mid x)$ while constraining its internal representation $\phi(x)$ to carry as little information about $z$ as possible. We quantify the demographic fairness of any classifier by the *fairness gap*

$$\Delta_{\text{eth}} \;=\; \max_{z} \mathrm{Acc}(z) \;-\; \min_{z} \mathrm{Acc}(z),$$

where $\mathrm{Acc}(z)$ is the test accuracy computed on the subset of utterances spoken by speakers of demographic group $z$. A smaller $\Delta_{\text{eth}}$ means the model treats each ethnic group more equally. We report the same metric for age-group and sex slices in Section V.

## IV-B Baseline Reproduction (RF and LR)

To establish a faithful comparison ground we re-implement the two baselines reported in Maltezou-Papastylianou et al. [1].

**Random Forest (RF).** We train a `RandomForestClassifier` (scikit-learn) with `n_estimators = 100`, `max_depth = 10`, and `random_state = 42`. The source paper uses 126 trees; we use 100 as a marginally more conservative configuration and verify in Section V-B that this small change does not move the headline accuracy. All 60 acoustic features are standardised with `StandardScaler` before fitting.

**Logistic Regression (LR).** We train a `LogisticRegression` model with the `liblinear` solver and `random_state = 42`. As LR is sensitive to feature scale, the same `StandardScaler` is applied. With fixed random state and a fully deterministic solver, the LR fit is reproducible across runs, which is reflected in its zero across-seed variance in Section V-B.

Both baselines use the joint $(\text{intent} \times \text{ethnicity})$ stratified $80/20$ split described in Section III-C. Stratifying jointly preserves the proportions of all six $(2 \text{ intents}) \times (3 \text{ ethnicities})$ cells in both train and test, which is essential for stable per-ethnicity accuracy estimates.

These models capture only the linear or tree-axis-aligned structure of the 60-dimensional acoustic space. They cannot represent interactions of the form "F0-variance × ethnicity → trustworthy", which motivates the deep models introduced next.

## IV-C Artificial Neural Network (ANN)

The ANN is a fully-connected feed-forward network with three hidden layers. Given an input $x \in \mathbb{R}^{60}$, the forward computation is

$$
\begin{aligned}
h_1 &= \text{Dropout}_{0.3}\bigl(\text{BN}\bigl(\text{ReLU}(W_1 x + b_1)\bigr)\bigr), \quad W_1 \in \mathbb{R}^{128 \times 60}, \\
h_2 &= \text{Dropout}_{0.3}\bigl(\text{BN}\bigl(\text{ReLU}(W_2 h_1 + b_2)\bigr)\bigr), \quad W_2 \in \mathbb{R}^{64 \times 128}, \\
h_3 &= \text{ReLU}(W_3 h_2 + b_3), \quad W_3 \in \mathbb{R}^{32 \times 64}, \\
\hat{y} &= \sigma(W_4 h_3 + b_4), \quad W_4 \in \mathbb{R}^{1 \times 32}.
\end{aligned}
$$

The trust head produces a calibrated probability $\hat{y} \in [0, 1]$ through the sigmoid $\sigma(\cdot)$. We optimise binary cross-entropy

$$\mathcal{L}_{\text{trust}}(\hat{y}, y) \;=\; -\bigl[\,y \log \hat{y} \,+\, (1 - y) \log(1 - \hat{y})\,\bigr]$$

with the Adam optimiser (learning rate $10^{-3}$), batch size 32 and 50 epochs. Fig. 4 shows the layer-by-layer architecture.

We chose three hidden layers as a balance between non-linear capacity and over-fitting risk: with only 921 training utterances and 60 input features, deeper or wider networks degraded test accuracy in preliminary experiments. Batch normalisation (BN) stabilises the loss surface in the small-data regime, and a 30% dropout rate after the two largest layers acts as a strong implicit regulariser. The 32-dimensional output of $h_3$ is the model's compressed representation of trustworthy-intent-relevant acoustics; we exploit this same bottleneck design in the DANN-Trust encoder below.

## IV-D One-Dimensional CNN

The 1D-CNN treats the 60-feature acoustic vector as a length-60 "signal" with a single channel and applies two stacked convolutions before a small dense head. The input is reshaped from $x \in \mathbb{R}^{60}$ to $X \in \mathbb{R}^{60 \times 1}$ and then

$$
\begin{aligned}
C_1 &= \text{Dropout}_{0.2}\bigl(\text{BN}\bigl(\text{ReLU}(\text{Conv1D}_{64,\,k=3}(X))\bigr)\bigr), \\
C_2 &= \text{ReLU}\bigl(\text{Conv1D}_{32,\,k=3}(C_1)\bigr), \\
h   &= \text{ReLU}\bigl(W \,\text{Flatten}(C_2) + b\bigr), \\
\hat{y} &= \sigma(W' h + b').
\end{aligned}
$$

The first convolutional layer applies 64 width-3 filters, the second applies 32; flattening yields a vector that is consumed by a hidden dense layer of width 64 before the sigmoid trust output. Training hyper-parameters (Adam $10^{-3}$, batch 32, 50 epochs, binary cross-entropy) match the ANN. Fig. 5 illustrates the architecture.

**A note on architectural motivation.** Unlike convolutions over a spectrogram or a time waveform, the 60 features in `Speech_dataset_characteristics.csv` are *not* arranged on a meaningful axis: column 5 is pitch, column 30 is shimmer, column 50 is long-term-average-spectrum slope. A width-3 convolutional kernel therefore slides across arbitrary feature triples and cannot exploit local structure the way it would on a spectrogram. We deliberately include the 1D-CNN as a *control* for the ANN: if a model whose primary inductive bias is locality nonetheless reaches similar accuracy and fairness as a fully-connected model, the gain we observe over RF / LR comes from depth and non-linearity in general, not from any specific architectural choice. The Section V results confirm this reading: ANN and CNN achieve overall accuracies within 1.1pp of each other.

## IV-E Domain-Adversarial Network (DANN-Trust)

Sections IV-C and IV-D show that depth and non-linearity already raise overall accuracy and substantially shrink the demographic gap. However, a strong ANN can still latch onto an ethnicity-correlated acoustic feature (for example, a long-term-spectrum pattern that differs between Black and South Asian speakers, as seen in Tables 5 and 6 of Maltezou-Papastylianou et al. [1]) and use it as a shortcut for "trustworthy". Two ANN runs with different random initialisations may pick *different* shortcuts, producing comparable mean accuracy but markedly different per-ethnicity gaps. The result is **fairness behaviour that is unpredictable across retrains**, which we observe directly in Section V (ANN gap std 1.10pp, CNN gap std 2.87pp).

We address this with a Domain-Adversarial Neural Network (Ganin et al., 2016) that explicitly suppresses ethnicity decoding from the shared representation. The architecture, shown in Fig. 6, has three sub-networks built on a shared encoder $G_f$:

* **Encoder** $G_f : \mathbb{R}^{60} \to \mathbb{R}^{32}$. Two dense layers (widths 128 and 32) with ReLU activations and batch normalisation; a 30% dropout follows the first layer. The 32-dimensional output is the trust-relevant representation $\phi(x) = G_f(x)$.
* **Trust head** $G_y : \mathbb{R}^{32} \to [0, 1]$. A 16-unit ReLU layer with 20% dropout, followed by a sigmoid output. This head predicts $\hat{y} = G_y(\phi(x))$.
* **Domain head** $G_d : \mathbb{R}^{32} \to \mathbb{R}^{K}$. A 16-unit ReLU layer with 20% dropout, followed by a softmax over $K = 3$ ethnicities. This head predicts $\hat{z} = G_d(\phi(x))$.
* **Gradient Reversal Layer (GRL)**, placed between $G_f$ and $G_d$. On the forward pass the GRL is the identity; on the backward pass it multiplies the upstream gradient by $-\lambda$. This is implemented in TensorFlow via `tf.custom_gradient` and verified by a unit test (forward equality, backward sign-flipped) before training.

The joint training objective combines a normal trust loss with a *negative-weighted* domain loss:

$$
\mathcal{L}_{\text{total}}(\theta_f, \theta_y, \theta_d) \;=\; \mathcal{L}_{\text{trust}}\bigl(G_y(G_f(x)),\, y\bigr) \;-\; \lambda \cdot \mathcal{L}_{\text{dom}}\bigl(G_d(G_f(x)),\, z\bigr),
$$

where $\mathcal{L}_{\text{trust}}$ is binary cross-entropy and $\mathcal{L}_{\text{dom}}$ is categorical cross-entropy. The minus sign is implemented implicitly by the GRL: the domain head still minimises $\mathcal{L}_{\text{dom}}$ (it learns to predict ethnicity as well as it can), but the gradient that flows back into $G_f$ is reversed, pushing the encoder to make $\phi(x)$ *less* informative about ethnicity. The encoder thus plays a min-max game against the domain head; at convergence, $G_d$ cannot reliably decode $z$ from $\phi(x)$, and any ethnicity-correlated shortcut has been removed.

The adversarial strength $\lambda$ follows the schedule recommended in (Ganin et al., 2016):

$$\lambda_p \;=\; \lambda_{\max} \cdot \left( \frac{2}{1 + \exp(-10\,p)} - 1 \right), \qquad p = \frac{\text{epoch}}{\text{total epochs}}.$$

With $\lambda_{\max} = 1$, $\lambda$ starts at $0$ (so the encoder first learns useful trust features without adversarial pressure) and asymptotes to $1$ by the final epoch (then the encoder is forced toward demographic invariance). We train end-to-end with Adam (learning rate $10^{-3}$), batch size 32, for 80 epochs — 30 more than the ANN/CNN, since adversarial training converges more slowly.

The architecture deliberately *matches* the ANN's bottleneck size (32 units): the only mechanical difference between ANN and DANN-Trust is the addition of the GRL-protected domain head and the corresponding loss term. Any difference observed in Section V can therefore be attributed to the adversarial constraint, not to extra capacity.

## IV-F Evaluation Metrics

For every model we report:

1. **Overall accuracy** on the held-out test set, and per-demographic accuracy on each ethnicity, age-group and sex slice.
2. **Precision, recall and F1-score** for both the Neutral and Trustworthy classes, to expose any asymmetry in which class the model finds harder.
3. **AUC-ROC** overall and per slice, providing a threshold-independent view of discriminability.
4. **Confusion matrices** for the overall test set and the per-ethnicity slices.
5. **Fairness gap** $\Delta_{\text{eth}} = \max_{z} \mathrm{Acc}(z) - \min_{z} \mathrm{Acc}(z)$, with analogous gaps $\Delta_{\text{age}}$ and $\Delta_{\text{sex}}$ for the other demographic axes. Lower is fairer.
6. **AUC fairness gap**, the same quantity computed on per-slice AUC, since accuracy gaps and AUC gaps can move independently.
7. **Probe-classifier accuracy on the DANN encoder.** As a sanity check on the adversarial mechanism, we extract the frozen encoder output $\phi(x)$ on the test set, train a fresh logistic-regression probe to predict $z$ from $\phi(x)$, and report its accuracy. For three ethnicities, chance is $33.3\%$; probe accuracy near chance indicates the adversary succeeded at scrubbing ethnicity from the representation, while probe accuracy substantially above chance indicates residual demographic leakage.

Each model is trained five times with seeds $\{42, 43, 44, 45, 46\}$ on the *same* fixed train/test partition (split seed $= 42$). This isolates *model variance* (how much a retrained model's behaviour shifts on the same data) from *split variance* (how much the train/test partition itself contributes), and lets us measure the variance-reduction claim of Section V-B directly: a model whose mean accuracy is matched by a competitor but whose seed-to-seed standard deviation is markedly smaller is a more reproducible — and therefore more deployable — model.
