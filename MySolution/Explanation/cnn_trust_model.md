# `cnn_trust_model.py` — Plain English Walkthrough

## The big idea: a "sliding window" version of the ANN

If [`ann_trust_model.py`](ann_trust_model.md) is the vanilla neural network — every neuron looks at all 60 features at once — this script is its more specialized cousin: a **1D Convolutional Neural Network (1D-CNN)**.

The core trick of a CNN is a **sliding window**. Imagine reading a book by looking at only three letters at a time, sliding your finger one position to the right after each look, and building up your understanding of the whole page from those little three-letter snapshots. That's a 1D convolution — a small filter that scans across the input looking for **local patterns** between adjacent positions.

CNNs were invented to look for patterns in things like:

- **Images** — nearby pixels form edges, corners, shapes.
- **Audio waveforms** — nearby samples form pitches, syllables.
- **Time series** — nearby timesteps form trends and cycles.

Here we're applying that same trick to the 60 acoustic voice measurements. But there's a catch — a really important one — that we'll come back to.

---

## An important caveat up front

The 60 features in the CSV (pitch mean, pitch std, jitter, shimmer, HNR, formant frequencies, …) are stored in **whatever order VoiceLab happened to output them**. There's no meaningful reason that "column 3" sits next to "column 4" — the ordering is essentially arbitrary.

But a CNN's whole design *assumes* neighbors are related. So this model is deliberately using a tool whose main superpower doesn't apply to this data.

**Why do it anyway?** Owen includes the CNN as a **control experiment** in the paper. The question is: if a model whose primary trick (locality) is mismatched to the data still matches the ANN's performance, then any improvement over the RF/LR baselines is coming from something *general* — like the depth and non-linearity of neural networks — rather than from clever use of feature ordering.

Think of it like giving a chef a knife when they need a rolling pin. If they still bake a great pie, you learn that talent matters more than the tool.

---

## Walking through the code, step by step

### Steps 1–5: Same as the ANN

Lines 1–41 are essentially identical to `ann_trust_model.py`:

- **Import libraries** (with one new one — `Conv1D` and `Flatten` from Keras).
- **Load the CSV** of voice recordings.
- **Encode `Speaker_Intent`** as 0/1.
- **Drop metadata columns** (filename, speaker ID, ethnicity, etc.) so the model can't cheat.
- **Fill missing values** with the column mean.
- **Split 80/20** with joint stratification on intent × ethnicity, `random_state=42`.
- **Standardize** with `StandardScaler` — fit on training data only, transform both halves.

For the full explanation of each of these steps, see [`ann_trust_model.md`](ann_trust_model.md). They exist here so this script stands alone and can be run without any dependencies on the other model scripts.

### Step 6 — Reshape the data for the CNN (lines 43–45)

```python
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0],  X_test_scaled.shape[1],  1)
```

This is the one bit of prep the CNN needs that the ANN didn't. Keras's `Conv1D` layer expects data with **three dimensions**:

```
(number_of_samples, length_of_sequence, number_of_channels)
```

For our data:

- `number_of_samples` = 921 (training recordings) or 231 (test recordings).
- `length_of_sequence` = 60 (the acoustic features, treated as a "sequence").
- `number_of_channels` = 1 (each position has just one number, not e.g. RGB triples).

That's why we do `.reshape(..., 1)` at the end — we're just wrapping every number in a 1-long list so the shape matches what `Conv1D` wants. The actual numbers don't change.

### Step 7 — Build the CNN (lines 47–56)

```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Four kinds of layer, in this order:

**1. `Conv1D(64, kernel_size=3, activation='relu')`** — the first sliding-window layer.

- `kernel_size=3` means each filter looks at **3 adjacent features at a time**. So filter #1 looks at features 1-2-3, then slides to 2-3-4, then 3-4-5, and so on, across all 60 positions.
- The `64` means there are **64 independent filters** working in parallel. Each one learns to detect a different kind of small pattern in the input. Think of it as 64 different tiny "pattern-detector" magnifying glasses all scanning the input at once.
- After scanning, this layer outputs 64 separate "pattern maps" — one per filter — each of length 58 (60 positions minus 2 lost at the edges from the sliding).

**2. `BatchNormalization() + Dropout(0.2)`** — same helpers as the ANN.

- BatchNorm keeps the numbers coming out of Conv1D at a stable scale.
- Dropout turns off 20% of the neurons during training to prevent overfitting.

**3. `Conv1D(32, kernel_size=3, activation='relu')`** — a **second** sliding window.

- This one scans across the *outputs* of the first Conv1D layer. Where the first layer detected simple local patterns, this second layer detects "combinations of the simple patterns" — i.e., patterns of patterns.
- 32 filters this time, again with `kernel_size=3`.
- Output length shrinks further to 56 (58 minus 2 more).

**4. `Flatten() + Dense(64) + Dense(1, sigmoid)`** — the final decision-maker.

- `Flatten()` = squash all the 2D "pattern maps" into one long flat list of numbers.
- `Dense(64, relu)` = a fully-connected layer that combines those flattened numbers into a 64-dimensional summary.
- `Dense(1, sigmoid)` = the final probability of "Trustworthy," between 0 and 1.

So the whole pipeline is:

```
60 features (as sequence)
      ↓ Conv1D(64, k=3)   ← scan for small local patterns
      ↓ Conv1D(32, k=3)   ← scan for patterns-of-patterns
      ↓ Flatten            ← lay it all flat
      ↓ Dense(64)          ← combine into a summary
      ↓ Dense(1, sigmoid)  ← output a probability
Trustworthy probability
```

Note there's **no Dropout after the first Conv1D** in this exact script (`Dropout(0.2)` sits between the two Conv1D layers). The regularization is lighter than the ANN's, which used `Dropout(0.3)` after two of its dense layers.

### Step 8 — Compile the model (line 59)

```python
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
```

Identical to the ANN. Adam optimizer, binary cross-entropy loss (correct choice for 0/1 targets), and both accuracy and AUC tracked as metrics.

### Step 9 — Train the model (lines 61–63)

```python
print("Training the 1D Convolutional Neural Network...")
model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)
```

Also identical to the ANN's training call: 50 passes over the training data (epochs), 32 recordings per weight update (batch size), and silent progress (`verbose=0`).

### Step 10 — Report the score (lines 65–69)

```python
loss, accuracy, auc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
```

Show the trained model the 20% of recordings it has never seen and print two numbers:

- **Accuracy** — fraction of test recordings labelled correctly. Target > 70%.
- **AUC** — how well the model ranks Trustworthy above Neutral. Target > 0.78.

---

## Why this model is scientifically interesting *even though* its main superpower doesn't fit the data

Recall the caveat from the top: the 60 features have no natural spatial ordering, so CNN's "look at neighbors" trick is mismatched with the data. That should hurt the CNN, right?

Two possible outcomes when you run this script:

- **If the CNN's accuracy matches the ANN's** → the accuracy improvement over RF/LR is coming from *general* neural-network properties (depth, non-linearity, batch normalization, dropout), not from any clever use of feature ordering. The paper can then argue that non-linear modeling — not any specific architectural trick — is what drives the fairness improvement.
- **If the CNN's accuracy is much worse than the ANN's** → the ANN was quietly benefiting from something the CNN can't access, and Owen would have to figure out what.

In the paper's Table 2, both the ANN and CNN sit within one point of each other on both accuracy and AUC. That's the CNN control doing its job: it confirms that Owen isn't secretly relying on some architectural fluke, which strengthens the DANN-Trust story that follows.

---

## What's NOT in this script (and why)

- **No multi-seed loop.** Runs once, prints one accuracy/AUC. If you rerun you'll get slightly different numbers because weight initialization and dropout are random. The real 25-seed experiment lives in `Analysis/compare_all_models.py`.
- **No per-ethnicity fairness gap.** Overall accuracy only. Fairness slicing (per-ethnicity, per-age, per-sex) happens in `compare_all_models.py`.
- **No pooling layers.** A "textbook" CNN would put a `MaxPooling1D` after each Conv1D to shrink the representation. This script skips pooling because the sequence is already short (60 positions), and pooling would throw away too much information. That's a reasonable choice for tabular-style data.
- **No adversarial fairness objective.** That's the extra layer added by `NewMethods/DANN_Trustworthy_Intent_Project/dann_trust_model.py`.

This file is the **minimal educational example** — you can read it in five minutes. The scientific machinery for the paper's actual claims is all in `MySolution/Analysis/`.

---

## In one sentence

**`cnn_trust_model.py` builds a two-Conv1D-layer network that slides small filters across the 60 acoustic features looking for local patterns — deliberately using a "neighbors matter" architecture on data whose feature ordering is arbitrary, as a control experiment to show that any performance gain over the baselines is coming from depth and non-linearity rather than from clever exploitation of feature layout.**
