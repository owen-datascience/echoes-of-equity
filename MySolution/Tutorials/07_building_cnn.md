# Tutorial 07 — Building a 1D CNN

Reference implementation: `../NewMethods/CNN_Trustworthy_Intent_Project/cnn_trust_model.py`.

You've built a Dense neural network (ANN). Now we'll build a **Convolutional Neural Network** — the kind of network that powers image recognition, speech recognition, and many medical AI systems. But applied to our *tabular* features rather than to a 2D image.

---

## What is a convolution, in 30 seconds?

A **convolution** is a small window that slides across the input, performing the same calculation at each position. It is looking for a *local pattern*.

### 2D version (image)

In image recognition, a 3×3 window slides across pixels, detecting edges, corners, textures.

```
   Input image:                Filter (kernel):
   ┌───────────────┐               ┌───┐
   │ . . . . . . . │               │ ▣ │ ── slides over the image
   │ . . . . . . . │               └───┘
   │ . . . . . . . │
   └───────────────┘
```

### 1D version (signal)

For 1-dimensional data — like our row of acoustic features, or a clip of raw audio — the window slides along a single axis:

```
   Input (a row of 58 numbers):
   [ x1  x2  x3  x4  x5  ... x58 ]
       └─┬─┘
        kernel of size 3 — slides right one step at a time
              └─┬─┘
                  └─┬─┘
                    ...
```

At each position, the filter combines 3 adjacent values into one output number. With 64 different filters, we get 64 output channels — each looking for a different local pattern.

### Why convolutional?

> A `Dense` layer treats every feature as completely independent. A `Conv1D` layer assumes nearby features might *relate* to each other and looks for those relationships.

In images, adjacent pixels obviously relate. In our acoustic features, the relationship is weaker (column order is somewhat arbitrary), but pitch-family features happen to be next to each other in the CSV, as are jitter-family and shimmer-family. So a 1D CNN can still find useful patterns.

---

## The plan

```
   Input: (58, 1)         ← 58 features, treated as a "signal" with 1 channel
        │
        ▼
   Conv1D(64 filters, kernel_size=3, relu)
        │
   BatchNormalization
        │
   Dropout(0.2)
        │
        ▼
   Conv1D(32 filters, kernel_size=3, relu)
        │
        ▼
   Flatten()                 ← Turn the 2D output into a 1D vector
        │
        ▼
   Dense(64, relu)
        │
        ▼
   Dense(1, sigmoid)
        │
        ▼
   Probability of "Trustworthy"
```

---

## Setup

Create the file:

```
MySolution/NewMethods/CNN_Trustworthy_Intent_Project/my_cnn.py
```

(Same folder as the other CNN copy of the CSV.)

---

## Step 1 — Imports

```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
```

Two new layer types:

- **`Conv1D`** — the 1D convolution layer.
- **`Flatten`** — converts the 2D output of the conv layers back into a 1D vector so a regular `Dense` layer can read it.

---

## Step 2 — Preprocessing (still the same)

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])

X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity',
                     'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Identical to the ANN tutorial.

---

## Step 3 — The CNN-specific twist: reshape the data

This is the one new preprocessing step.

```python
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0],  X_test_scaled.shape[1], 1)
```

### Why?

A `Conv1D` layer expects 3D input: `(batch, length, channels)`.

- **batch** = number of samples
- **length** = length of the signal (here, 58 features in a row)
- **channels** = how many parallel signals (here, just 1 — we have a single value per feature)

For comparison, an image CNN sees `(batch, height, width, 3)` where 3 is the RGB channel count.

After reshape, each sample is a `(58, 1)` array — a "signal" 58 numbers long with 1 channel. Same data, just a different shape so Keras knows how to treat it.

---

## Step 4 — Build the CNN

```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu',
           input_shape=(X_train_scaled.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(32, kernel_size=3, activation='relu'),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
```

Let's understand the new pieces.

### `Conv1D(64, kernel_size=3, activation='relu')`

- **64 filters** — 64 different "local pattern detectors", each independently learnable.
- **`kernel_size=3`** — each filter looks at 3 adjacent features at a time.
- **activation='relu'** — same nonlinearity we use everywhere.

By default, the filter slides one step at a time (`strides=1`) and doesn't pad the edges (`padding='valid'`). So a length-58 input becomes length `58 - 3 + 1 = 56`.

After the first `Conv1D`, the shape is `(batch, 56, 64)` — 56 sliding positions, 64 channels.

### Second `Conv1D` — finding patterns of patterns

```python
Conv1D(32, kernel_size=3, activation='relu')
```

The second conv layer looks at 3-position windows of the *first layer's output*. So it's not looking at the raw data — it's looking at combinations of patterns the first layer detected. Each filter combination → 32 outputs, each `56 - 3 + 1 = 54` long. New shape: `(batch, 54, 32)`.

This idea — earlier layers learn primitive patterns, later layers combine them into higher-level patterns — is one of the most beautiful things about deep learning.

### `Flatten()`

Up to here, the data is 2D per sample: 54 positions × 32 channels = 1,728 numbers per sample. `Flatten()` literally lays them out in a single row so a regular `Dense` layer can process them.

### Final dense layers

```python
Dense(64, activation='relu'),
Dense(1, activation='sigmoid'),
```

The same two-step "interpret + decide" pattern from the ANN.

---

## Step 5 — Compile + Train + Evaluate

These three steps are *literally identical* to the ANN. Once you've learned the API, you can build different model architectures by swapping out the layer stack — the rest is boilerplate.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Training the 1D Convolutional Neural Network...")
model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)

loss, accuracy, auc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
```

Notice we pass `X_train_cnn` (the reshaped 3D version), not `X_train_scaled`.

Run it:

```bash
python my_cnn.py
```

Expected output:

```
Training the 1D Convolutional Neural Network...
--- RESULTS ---
Accuracy: 71.55% (Target: >70%)
AUC Score: 0.77 (Target: >0.78)
```

For *this* dataset, the CNN is not dramatically better than the ANN. That's because our features aren't strongly sequential — they're more like 58 independent measurements. CNNs really shine on data where order matters a lot (raw audio waveforms, time series, images).

But you have now built one, and the same code pattern scales up to all those richer applications.

---

## Visualize the architecture

```python
model.summary()
```

You'll see something like:

```
Layer (type)         Output Shape       Param #
=================================================
conv1d (Conv1D)      (None, 56, 64)     256
batch_normalization  (None, 56, 64)     256
dropout              (None, 56, 64)     0
conv1d_1 (Conv1D)    (None, 54, 32)     6,176
flatten              (None, 1728)       0
dense (Dense)        (None, 64)         110,656
dense_1 (Dense)      (None, 1)          65
=================================================
Total params: 117,409
```

The bulk of the parameters live in the `Dense(64)` layer right after `Flatten`. That's normal — fully connected layers have many connections by definition.

---

## Hyperparameters to play with

| Knob | Default | Try | Why |
|------|---------|-----|-----|
| `kernel_size` | 3 | 5, 7 | Bigger window → larger receptive field but fewer output positions |
| Number of filters | 64 → 32 | 32 → 16 | Smaller network, less overfitting |
| Add a third Conv1D | — | `Conv1D(16, 3, 'relu')` before `Flatten` | Deeper feature extraction |
| `strides=2` in Conv1D | 1 | 2 | Halves the output length (faster, less detail) |
| Add `MaxPooling1D(2)` after a conv | — | Reduces length by 2 | Classic CNN downsampling |

---

## When to choose CNN vs ANN

| Situation | Pick |
|-----------|------|
| Plain tabular data, columns are unrelated | ANN (Dense) |
| Sequential / signal / time series | 1D CNN |
| Images | 2D CNN |
| Long-range dependencies (text, music) | Transformer / RNN |

For our *engineered* acoustic features the ANN is arguably more natural. But if we had the raw audio waveform, a 1D CNN would crush an ANN by detecting local sound patterns directly.

---

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgot to reshape to 3D | `ValueError: Input 0 of layer ... is incompatible` | Reshape `(samples, features, 1)` |
| Forgot `Flatten()` before `Dense` | Shape mismatch error before the dense layer | Add `Flatten()` |
| `kernel_size` ≥ number of features | Conv output has length 0 or negative | Use a small kernel |
| Passed scaled 2D data to `model.fit` | `Input shape (None, 58)` won't match `(None, 58, 1)` | Pass the reshaped `X_train_cnn` |

---

## Self-check

1. What does a 1D convolution do? Use the phrase "sliding window" in your answer.
2. Why do CNNs need 3D input?
3. What does `Flatten` do, and why is it needed before `Dense`?
4. Why might a CNN not outperform an ANN on this particular dataset?

You now have all four models working. Head to **[Tutorial 08 — Comparing Models](08_comparing_models.md)** to compare them side by side and discuss where to go next.
