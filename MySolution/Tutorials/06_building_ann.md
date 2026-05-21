# Tutorial 06 — Building an ANN

Reference implementation: `../NewMethods/ANN_Trustworthy_Intent_Project/ann_trust_model.py`.

Time to put Tutorial 05's vocabulary into code. You will build a real deep neural network with TensorFlow.

---

## The plan

We are building this network:

```
   Input (58 features)
        │
        ▼
   Dense(128, relu)              ← Layer 1: 128 neurons
        │
   BatchNormalization
        │
   Dropout(0.3)
        │
        ▼
   Dense(64, relu)               ← Layer 2: 64 neurons
        │
   BatchNormalization
        │
   Dropout(0.3)
        │
        ▼
   Dense(32, relu)               ← Layer 3: 32 neurons
        │
        ▼
   Dense(1, sigmoid)             ← Output: 1 number, between 0 and 1
        │
        ▼
   Probability of "Trustworthy"
```

Notice how the layers get **narrower** going down (128 → 64 → 32 → 1). This is a common shape — early layers cast a wide net for raw patterns; later layers focus and combine.

---

## Set up your workspace

If you haven't yet, install TensorFlow:

```bash
pip install tensorflow
```

> On Windows, the install can take a few minutes and is a hefty download (~400 MB). If you have a slow machine, this is the model that will train slowest in this tutorial series — but each individual epoch is still fast.

Create a new file:

```
MySolution/NewMethods/ANN_Trustworthy_Intent_Project/my_ann.py
```

(Make sure the CSV is in the same folder; the reference project keeps a copy alongside each script.)

---

## Step 1 — Imports

```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
```

Three new pieces:

- `tensorflow as tf` — the library.
- `Sequential` — the model class for "stack these layers in order".
- `Dense, Dropout, BatchNormalization` — the layer types you met in Tutorial 05.

---

## Step 2 — Preprocessing (same as before)

```python
# Load
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

# Encode label
encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])

# Features and target
X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity',
                     'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']

# Fill missing
X = X.fillna(X.mean())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale (critical for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

Identical to Tutorials 03 and 04. Notice we're not even bothering with the `np.inf` replace step — `fillna(X.mean())` is usually enough. (If you hit an error, add it back.)

---

## Step 3 — Build the model

This is the heart of the tutorial.

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid'),
])
```

Let's read this top-to-bottom.

### `Sequential([...])`

The simplest model type in Keras. It runs the layers in the order you list them, top to bottom.

### `Dense(128, activation='relu', input_shape=(...))`

- **128** = number of neurons in this layer.
- **activation='relu'** = the activation function from Tutorial 05.
- **input_shape=(58,)** = one tuple telling the network what *one* sample looks like. We only specify this on the *first* layer; Keras figures out the rest.

Why a comma? `(58,)` is Python's way of writing a 1-element tuple. It says "the input is a 1-dimensional vector with 58 elements". This is `(number_of_features,)`, which we compute as `X_train_scaled.shape[1]`.

### `BatchNormalization()`

Rescales the activations leaving the previous layer to roughly mean 0, std 1. Speeds up training and stabilizes it.

### `Dropout(0.3)`

During training, randomly zero out 30% of this layer's outputs. Prevents memorization.

### Two more hidden layers

```python
Dense(64, activation='relu'),
BatchNormalization(),
Dropout(0.3),

Dense(32, activation='relu'),
```

Same pattern, but with fewer neurons each time. The third layer skips dropout because it's already small.

### The output layer

```python
Dense(1, activation='sigmoid')
```

One neuron with a sigmoid activation. The output is a single number between 0 and 1 — the probability that the input belongs to class 1 (Trustworthy).

---

## Step 4 — Compile

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
```

- **`optimizer='adam'`** — the algorithm that adjusts weights. Adam is the go-to default.
- **`loss='binary_crossentropy'`** — the right loss function for binary classification with a sigmoid output. (For multi-class you'd use `categorical_crossentropy`.)
- **`metrics=[...]`** — what to report during training and evaluation. We watch both `accuracy` and `AUC`.

> **Loss vs metric:** the optimizer minimizes the *loss* directly. Metrics are just for *us* to watch. You can have many metrics, but only one loss.

---

## Step 5 — Train

```python
print("Training the Deep Neural Network... Please wait.")
model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    verbose=0     # set to 1 if you want to see per-epoch output
)
```

- **`epochs=50`** — sweep through the entire training set 50 times.
- **`batch_size=32`** — process 32 rows per training step.
- **`verbose=0`** — quiet mode. If you flip it to `verbose=1`, you'll see a progress bar with the loss going down — pretty satisfying.

> Try `verbose=1` the first time so you can watch the model learn. Then switch back to `verbose=0` for cleaner output.

---

## Step 6 — Evaluate

```python
loss, accuracy, auc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
```

`evaluate` returns the same metrics in the same order you compiled them in. Since we compiled with `loss + ['accuracy', 'auc']`, it returns three numbers.

Run it:

```bash
python my_ann.py
```

Expected output (yours will vary by a few points):

```
Training the Deep Neural Network... Please wait.
--- RESULTS ---
Accuracy: 72.38% (Target: >70%)
AUC Score: 0.79 (Target: >0.78)
```

You just trained a neural network from scratch. Congratulations.

---

## What just happened, in slow motion

For each of 50 epochs, for each batch of 32 rows:

1. The network turned 58 acoustic features → 128 hidden values → 64 → 32 → 1 probability.
2. It compared the probability to the true label (`binary_crossentropy`).
3. It computed gradients backward through every weight.
4. Adam used those gradients to nudge weights toward better predictions.

By the end, the network's millions of tiny weights have collectively shaped themselves to recognize "trustworthy" voice patterns.

---

## Bonus — see the network's architecture

Add this right after building the model:

```python
model.summary()
```

You'll see a table like:

```
Layer (type)                 Output Shape       Param #
=====================================================
dense (Dense)                (None, 128)        7,552
batch_normalization          (None, 128)        512
dropout                      (None, 128)        0
dense_1 (Dense)              (None, 64)         8,256
...
Total params: 18,977
Trainable params: 18,721
Non-trainable params: 256
```

Almost 19,000 learnable numbers. That's a lot of weights for 1,200 rows of data — which is exactly why we needed dropout and batch-norm to prevent overfitting.

---

## Experiment ideas

Don't just take the model as given. Change things and see what happens:

| Try | Expected effect |
|-----|-----------------|
| `epochs=5` | Underfit — too little training |
| `epochs=200` | Maybe overfit — training accuracy creeps up, test accuracy drops |
| Remove all `Dropout` layers | Overfit |
| Bigger layers: `Dense(512, ...)` | More params, slower, possibly overfits |
| Smaller layers: `Dense(8, ...)` | Underfit, but trains very fast |
| `activation='tanh'` instead of relu | Slightly different training dynamics |

For each experiment, run the script, write down the test accuracy and AUC, and try to *explain* what you see. That is exactly what real ML practitioners do all day.

---

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Forgot `input_shape` on first layer | "You must specify input_shape" error | Add `input_shape=(X.shape[1],)` |
| Used `categorical_crossentropy` on binary task | Loss won't decrease | Use `binary_crossentropy` |
| Final layer has `activation='relu'` | Output isn't a probability; AUC is broken | Use `sigmoid` for binary |
| Forgot to scale features | Training stalls or diverges | `StandardScaler` ALWAYS for NN |
| `batch_size=1` | Training is extremely slow and noisy | Use 16, 32, or 64 |

---

## Self-check

1. Why does the final layer have `Dense(1, activation='sigmoid')`?
2. What does `Dropout(0.3)` do during training? At prediction time?
3. What is the role of the optimizer?
4. If your training loss is going down but your test accuracy stops improving, what's happening?

When ready, head to **[Tutorial 07 — Building a CNN](07_building_cnn.md)** — a different network type that "scans" features for local patterns.
