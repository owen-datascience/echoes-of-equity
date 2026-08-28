# `dann_trust_model.py` — Plain English Walkthrough

## The big idea: a neural network with an internal tug-of-war

The [ANN](ann_trust_model.md) predicts trustworthiness from voice. The [CNN](cnn_trust_model.md) does the same with a sliding-window trick. Both work — but neither of them **cares about fairness**. They'll happily use ethnicity-related acoustic patterns as shortcuts if those patterns help predict "Trustworthy."

`dann_trust_model.py` is Owen's answer to that. It builds a neural network with an **adversary built inside it** — a second head whose only job is to predict speaker ethnicity from the encoder's output. Then, through a clever gradient trick, the encoder is *punished* whenever the ethnicity head does well.

The result is a tug-of-war inside a single model:

```
                             Trust head:    "Is this person trustworthy?"
                                            (pulls encoder toward being useful)
                                       ↑
    60 features → [encoder] ───────────┤
                                       ↓
                             Domain head:   "Which ethnicity is this?"
                                            (pulls encoder toward being useless
                                             for this — via a sign flip!)
```

If the tug-of-war balances well, the encoder learns to represent voices in a way that's **useful for trust prediction but useless for ethnicity prediction**. That should force the trust head to base its answers on ethnicity-neutral acoustic cues.

The technique comes from a 2016 paper by Ganin and colleagues, originally invented for a different problem (domain adaptation). This script is DANN adapted for **fairness**.

---

## Walking through the code, section by section

### Section 1 — Reproducibility (lines 22–37)

```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Because neural networks contain many sources of randomness (weight initialization, dropout, mini-batch shuffling), running the same script twice can give slightly different answers. Setting seeds pins the randomness so this script produces the *same* results on every run — critical for reporting numbers that others can reproduce.

### Section 2 — The Gradient Reversal Layer, the heart of the trick (lines 39–72)

**The math bit** (lines 43–49):

```python
@tf.custom_gradient
def _gradient_reversal_op(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, None
    return tf.identity(x), grad
```

This is a **custom-gradient operation** in TensorFlow. Two behaviors are defined separately:

- **Forward pass** (`tf.identity(x)`): output = input, unchanged. Passes information through like an open door.
- **Backward pass** (`grad(dy)`): when the correction signal `dy` comes back during training, flip its sign and scale it by `lambda_`.

That sign flip is the whole trick. Normally, during training, correction signals tell earlier layers "adjust yourselves to make the downstream loss smaller." A flipped sign says "adjust yourselves to make the downstream loss *bigger*." Which means: the encoder is actively trained to *sabotage* the ethnicity head. Meanwhile the ethnicity head itself is still trained normally to predict ethnicity as well as it can. Two-player game.

**The Keras layer wrapper** (lines 52–72):

```python
class GradientReversalLayer(layers.Layer):
    def __init__(self, initial_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = tf.Variable(
            float(initial_lambda), trainable=False, dtype=tf.float32,
            name="grl_lambda",
        )

    def call(self, x):
        return _gradient_reversal_op(x, self.lambda_)

    def set_lambda(self, value):
        self.lambda_.assign(float(value))
```

Keras is TensorFlow's high-level building-block API. Wrapping the custom-gradient op in a `Layer` class lets it slot into a Keras model like any other layer.

`self.lambda_` is a `tf.Variable` — but marked `trainable=False`, meaning the model never learns it. It's a **dial** you can turn from outside (via `set_lambda`) to change how hard the encoder is pushed to forget ethnicity.

### Section 3 — Ramping up the tug-of-war gradually (lines 75–93)

```python
class LambdaScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.total_epochs
        new_lambda = self.lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        self.grl_layer.set_lambda(new_lambda)
```

Why not just set `lambda = 1.0` from the start? Because at the very beginning of training, the encoder is *random* — it's not yet good at anything. If you immediately push it to "forget ethnicity," it might collapse before learning anything useful.

Ganin's recipe is: **start `lambda` near 0, ramp up smoothly to `lambda_max`**. The formula `2 / (1 + exp(-10p)) - 1` (where `p` = current epoch / total epochs) is a scaled sigmoid that goes:

- Early training (`p ≈ 0`): `lambda ≈ 0`. The encoder freely learns to predict trust. Adversary is barely engaged.
- Mid training (`p ≈ 0.5`): `lambda ≈ 0.5`. The tug-of-war ramps up.
- Late training (`p ≈ 1`): `lambda ≈ lambda_max`. Full adversarial pressure. Encoder is now being aggressively pushed to strip ethnicity from its output.

`Callback` is a Keras hook that Keras runs at specific moments — here, at the *start of every epoch*. So the value of `lambda` slides smoothly upward over the 80 epochs.

### Section 4 — Load the data with TWO labels (lines 96–127)

```python
trust_encoder = LabelEncoder()
ethn_encoder = LabelEncoder()
df["Speaker_Intent_enc"] = trust_encoder.fit_transform(df["Speaker_Intent"])
df["Speaker_Ethnicity_enc"] = ethn_encoder.fit_transform(df["Speaker_Ethnicity"])
```

Unlike the ANN and CNN, DANN needs **two labels** per recording:

- `y_trust`: 0/1 (Neutral/Trustworthy) — the main task.
- `y_ethn`: 0/1/2 (White/Black/South_Asian) — the adversarial task.

Both are computed here up front. The 60 acoustic features are extracted the same way as the other models (drop metadata, replace ±inf with NaN, fill NaN with column mean).

### Section 5 — The joint-stratified split (lines 129–157)

```python
strata = (y_trust.astype(int) * 10 + y_ethn).astype(int)
train_test_split(X.values, y_trust, y_ethn, ethn_str,
                 test_size=0.2, stratify=strata, random_state=SEED)
```

Same trick as the ANN and CNN: instead of stratifying only by intent (which would let the test set's ethnicity mix drift), the split preserves the proportions of all six `intent × ethnicity` cells. This is doubly important here because DANN's fairness numbers *are the point* — if the test set's per-ethnicity counts are lopsided, the fairness gap becomes noisy and misleading.

The script also prints a per-cell breakdown so a reviewer can eyeball that the balance actually worked.

### Section 6 — Assemble the two-headed model (lines 163–199)

```python
def build_dann_model(input_dim, num_domains):
    inputs = layers.Input(shape=(input_dim,))

    # Encoder G_f: shared brain
    h = layers.Dense(128, activation="relu")(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(32, activation="relu")(h)
    encoder_output = layers.BatchNormalization()(h)

    # Trust head G_y: predicts trustworthiness
    t = layers.Dense(16, activation="relu")(encoder_output)
    t = layers.Dropout(0.2)(t)
    trust_out = layers.Dense(1, activation="sigmoid")(t)

    # Domain head G_d: predicts ethnicity (via GRL)
    grl = GradientReversalLayer(initial_lambda=0.0)
    d = grl(encoder_output)
    d = layers.Dense(16, activation="relu")(d)
    d = layers.Dropout(0.2)(d)
    domain_out = layers.Dense(num_domains, activation="softmax")(d)

    model = Model(inputs=inputs, outputs=[trust_out, domain_out])
```

Three parts glued together, using Keras's functional API (each layer is called on the previous layer's output):

- **Encoder** (`128 → 32`): produces a 32-number "summary" of each voice recording. This is the shared brain that both heads use.
- **Trust head** (`16 → 1 sigmoid`): a small network that outputs the probability of trustworthy intent.
- **Domain head** (`GRL → 16 → 3 softmax`): another small network that tries to guess which of the 3 ethnicities the speaker belongs to. The GRL sits *between* the encoder and this head, silently flipping the correction signal.

Two extras stashed on the model (lines 191–194):

```python
model.grl_layer = grl              # so LambdaScheduler can reach it
model.encoder_only = encoder_only  # a separate view of just the encoder
```

The `encoder_only` submodel shares weights with the main model but exposes just the encoder's 32-number output. Later, we use it to run the "probe" test in Section 11.

### Section 7 — Compile with two losses (lines 201–218)

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "trust_output": "binary_crossentropy",
        "domain_output": "sparse_categorical_crossentropy",
    },
    loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    metrics={...},
)
```

Because the model has **two outputs**, Keras needs two loss functions — one per head:

- `binary_crossentropy` for trust (2 classes, sigmoid output).
- `sparse_categorical_crossentropy` for ethnicity (3 classes, softmax output).

`loss_weights` says "count both losses equally when adding them up." From Keras's viewpoint, the model has just two normal losses that get added together. **Keras has no idea that the domain-head loss is secretly adversarial** — the GRL takes care of that entirely behind the scenes, at the gradient level.

### Section 8 — Train, with the tug-of-war ramping up (lines 220–232)

```python
history = model.fit(
    X_train_scaled,
    {"trust_output": y_trust_train, "domain_output": y_ethn_train},
    epochs=TOTAL_EPOCHS,     # 80
    batch_size=32,
    verbose=0,
    callbacks=[LambdaScheduler(model.grl_layer, TOTAL_EPOCHS, lambda_max=1.0)],
)
```

Notice `y=` is a **dictionary** — one target array per output head. At each training step, both heads produce predictions, both losses are computed, and the combined gradient flows backward. The GRL flips its slice of the gradient on the way back.

The `LambdaScheduler` callback runs at the start of each epoch, sliding `lambda` up along the Ganin sigmoid from ~0 to 1.0 over 80 epochs.

DANN is trained for **80 epochs** (vs the ANN/CNN's 50) because the adversarial training is harder — the two heads can pull in opposite directions and it takes more time for them to reach a balance.

### Section 9 — Overall evaluation on the trust task (lines 234–249)

```python
trust_probs, _ = model.predict(X_test_scaled, verbose=0)
trust_preds = (trust_probs >= 0.5).astype(np.int32)
acc = accuracy_score(y_trust_test, trust_preds)
auc = roc_auc_score(y_trust_test, trust_probs)
```

The model returns two outputs (trust probabilities and domain probabilities). We only care about trust here, so we ignore the second one with `_`.

Two headline numbers:

- **Accuracy** target > 70% — must at least tie the RF/LR baselines to be useful.
- **AUC** target > 0.78 — a threshold-free view of discriminative quality.

If DANN sacrifices too much trust accuracy to be fair, it's not a viable model. The paper's argument depends on DANN reaching competitive accuracy while also reducing the fairness gap.

### Section 10 — Per-ethnicity fairness (lines 251–273)

```python
for ethn_name in ethn_encoder.classes_:
    mask = (ethn_str_test == ethn_name)
    g_acc = accuracy_score(y_trust_test[mask], trust_preds[mask])
    g_auc = roc_auc_score(y_trust_test[mask], trust_probs[mask])
```

For each ethnicity separately, compute accuracy and AUC using only that group's test recordings. Then:

```python
gap_pp = (max(per_ethn_acc.values()) - min(per_ethn_acc.values())) * 100
```

The **fairness gap** — the difference between the best-performing group and the worst-performing group — is the single number that summarizes disparity. A smaller gap means the model works more evenly across groups. The comparison reference (5pp under the original TIS paper's Random Forest) is printed for context.

### Section 11 — The probe test: did the adversary actually work? (lines 275–289)

This is the **check on the check** — the most philosophically important part of the script.

The adversarial training claims to be forcing the encoder to strip out ethnicity. But does that actually happen? Or does the encoder just learn to fool the built-in ethnicity head while still leaking ethnicity in some other way?

To find out, we train a **fresh classifier** — a plain logistic regression — on the frozen encoder's outputs, and ask it to predict ethnicity.

```python
enc_train = model.encoder_only.predict(X_train_scaled, verbose=0)  # 32-dim summaries
enc_test = model.encoder_only.predict(X_test_scaled, verbose=0)

probe = LogisticRegression(max_iter=2000, random_state=SEED)
probe.fit(enc_train, y_ethn_train)
probe_acc = accuracy_score(y_ethn_test, probe.predict(enc_test))
```

Interpretation:

- **Probe accuracy near 33.3%** (= 1/3, pure guessing for 3 ethnicities) → the encoder really has stripped ethnicity. Adversary won.
- **Probe accuracy much higher than 33.3%** → ethnicity is still recoverable from the encoder's output, just not by the model's own domain head. Adversary only partially won.

Owen's paper reports probe accuracy around 53.7% — well above chance. That's the honest verdict: the adversary reduces but doesn't eliminate ethnicity leakage. The paper's discussion section (§V-B) tries to reframe this — "DANN reduces but doesn't fully remove demographic information" — and this probe number is the evidence for that admission.

---

## Why this is Owen's headline contribution

The ANN and CNN are strong baselines but they have no fairness objective. The RF and LR baselines from the original TIS paper had the same problem — plus they were shallower and less accurate. DANN is Owen's attempt to close both gaps at once:

- Match the ANN/CNN on accuracy (comparable non-linear modeling capacity).
- **Beat everyone** on demographic balance (adversarial pressure on the encoder).
- Provide a *diagnostic* (the probe) that quantifies just how much demographic information is actually being removed.

Whether DANN succeeds is what the multi-seed comparison in `Analysis/compare_all_models.py` measures. This script builds and trains the model; the harness measures whether it deserves the win.

---

## What's NOT in this script (and why)

- **No multi-seed loop.** Trains once at `SEED=42` and prints one accuracy/AUC/gap/probe number. The 25-seed statistical experiment lives in `Analysis/compare_all_models.py` (main comparison) and `Analysis/run_dann_ablation.py` (ablation).
- **No comparison against the ANN/CNN.** This script trains DANN in isolation. Head-to-head comparison is done in `compare_all_models.py`.
- **No ablation of the GRL / BN / domain head.** Removing components one at a time to see which ones matter is what `run_dann_ablation.py` does. See [`run_dann_ablation.md`](run_dann_ablation.md).

Think of this file as the **canonical single-run version** of DANN-Trust — clean, complete, comment-rich, and safe to read top-to-bottom to understand exactly how the adversarial mechanism works. The scientific rigor (many seeds, statistical tests, ablations) is layered on top of this same architecture in the `Analysis/` folder.

---

## In one sentence

**`dann_trust_model.py` builds a two-headed neural network — a shared encoder feeding both a trustworthiness predictor and an ethnicity predictor — where a Gradient Reversal Layer secretly flips the sign of the ethnicity head's correction signal, forcing the encoder to become useful for predicting trust but useless for predicting ethnicity, and then verifies the result with a probe classifier trained on the frozen encoder's outputs.**
