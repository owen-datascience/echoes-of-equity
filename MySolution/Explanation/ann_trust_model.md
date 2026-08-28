# `ann_trust_model.py` — Plain English Walkthrough

## The big idea: build a brain that recognizes "sincere" voices

When you talk on the phone, your voice carries more than just words — it carries **pitch**, **loudness**, **breathiness**, and dozens of other properties. Even hearing the single word "hello" is often enough for a listener to form a snap judgment about whether the speaker sounds trustworthy.

This script trains a small **artificial neural network (ANN)** — a math model loosely inspired by neurons in a brain — to guess whether a recorded speaker was *trying* to sound trustworthy or was just talking normally. The input is 60 numerical measurements of the voice; the output is a single probability between 0 and 1.

Think of it as teaching a music student to tell a **major chord** from a **minor chord**. You don't tell them the rules directly — you just play thousands of chords labelled "major" or "minor" and let them figure out the pattern. That's what this script does with voices.

---

## Walking through the code, step by step

### Step 1 — Import the tools (lines 5–14)

```python
import pandas as pd                                    # spreadsheet-like data handling
import numpy as np                                     # numerical math
import tensorflow as tf                                # AI framework
from tensorflow.keras.models import Sequential         # stack model layers linearly
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split   # split data into train/test
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
```

Two ecosystems doing different jobs:

- **TensorFlow / Keras** builds and trains the neural network.
- **scikit-learn** (`sklearn`) handles the boring but critical preparation work: splitting data, scaling numbers, and turning word labels into numeric ones.

### Step 2 — Load the voice dataset (lines 16–19)

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)
```

The dataset is a CSV with one row per voice recording. Every row has:

- **Metadata** — filename, speaker ID, ethnicity, age group, sex, and the label (Neutral or Trustworthy).
- **60 acoustic measurements** — things like average pitch, jitter (how shaky the pitch is), shimmer (how uneven the volume is), and harmonics-to-noise ratio (how clean the voice is).

Using `script_dir + csv_path` (rather than just `'Speech_dataset_characteristics.csv'`) means the script works no matter what folder you're standing in when you run it — a small robustness trick.

### Step 3 — Turn words into numbers (lines 21–24)

```python
encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])
```

Computers can't do math on the word "Trustworthy." So `LabelEncoder` maps:

- `"Neutral"` → `0`
- `"Trustworthy"` → `1`

Now the target column is 0s and 1s that the neural network can chew on.

### Step 4 — Pick the input features (lines 26–30)

```python
X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                    'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']
```

- `X` (uppercase) = the 60 acoustic measurements — what the model *looks at*.
- `y` (lowercase) = the intent label — what the model *tries to guess*.

Notice what got dropped: filename, speaker ID, ethnicity, age, sex. Feeding those into the model would be **cheating** — the model could learn "any voice from speaker #47 is trustworthy" instead of learning what trustworthy voices sound like. This is called **preventing data leakage**.

### Step 5 — Clean up missing numbers (line 33)

```python
X = X.fillna(X.mean())
```

Some voice measurements might be blank if VoiceLab couldn't compute them for a particular recording (say, because the audio was too quiet). We fill in blanks with the **column average** so the neural network doesn't crash on missing values. It's a rough fix — not perfect, but standard.

### Step 6 — Split into training and testing (lines 35–41)

```python
_eth_codes = LabelEncoder().fit_transform(df['Speaker_Ethnicity'])
_strata = y.values * 10 + _eth_codes            # 6 combined labels: 2 intents x 3 ethnicities
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=_strata, random_state=42)
```

Two important things happening here:

**1. The 80/20 split.** 80% of the recordings go into training (the model gets to learn from them), 20% into testing (the model *never* sees them until we grade its work). Without this split, you can't tell if the model is really learning or just memorizing.

**2. Joint stratification.** The `stratify=_strata` argument makes sure both halves contain the same *proportions* of each `(intent × ethnicity)` combination. There are 6 cells:

```
Neutral × White    |   Trustworthy × White
Neutral × Black    |   Trustworthy × Black
Neutral × S.Asian  |   Trustworthy × S.Asian
```

Without stratification, a random split might accidentally put 90% of the South Asian speakers into the training set. Then the test set's per-ethnicity accuracy for South Asian speakers would be measured on almost no data, and the numbers would be meaningless. `random_state=42` makes the split reproducible — anyone rerunning this gets the exact same split.

### Step 7 — Rescale the numbers (lines 43–46)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The 60 features are on *wildly* different scales — pitch might be measured in hundreds of Hertz, jitter might be a tiny fraction like 0.003. Neural networks train much better when all features live on a similar scale.

`StandardScaler` transforms each feature to have mean 0 and standard deviation 1. Key subtlety: `fit_transform` is called on *training* data only — the scaler learns the mean and std from the training set, and then just `transform`s the test set with those same numbers. This mimics a real deployment where you can't peek at future data.

### Step 8 — Build the neural network (lines 48–58)

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

Think of this as a **funnel**:

```
60 features  →  Dense(128)  →  Dense(64)  →  Dense(32)  →  Dense(1)  →  probability
```

Every arrow is a **layer** of artificial neurons. Each neuron combines the previous layer's outputs with weights it learns during training, then passes the result through a "shape" function (`relu` for hidden layers, `sigmoid` for the output).

- **`Dense(N)`** = a fully-connected layer with `N` neurons. 128 → 64 → 32 gradually squeezes 60 measurements down into a small, meaningful summary.
- **`activation='relu'`** = "Rectified Linear Unit" — if a neuron's input is negative, output 0; otherwise output the input as-is. Simple and effective, and what makes neural networks non-linear.
- **`activation='sigmoid'`** = squashes the final number into 0..1, so it reads like a probability of "Trustworthy."

Two important helper layers appear between the Dense layers:

- **`BatchNormalization()`** = automatically re-centers and re-scales each layer's outputs during training. Helps training stay stable and converge faster. Kind of like a thermostat that keeps the model at a comfortable operating temperature.
- **`Dropout(0.3)`** = randomly turns off 30% of the neurons during each training step. Sounds crazy, but it forces the network to *not rely too heavily on any one neuron*, which prevents **overfitting** (memorizing the training set instead of learning general patterns). Think of it as making a study group do exam prep without one member each night — everyone has to actually understand the material.

The final `Dense(1, activation='sigmoid')` outputs one number between 0 and 1. Above 0.5 → "predict Trustworthy." Below 0.5 → "predict Neutral."

### Step 9 — Set the training rules (line 61)

```python
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
```

Three settings the model needs before training:

- **Optimizer `adam`** = the algorithm that adjusts the weights to reduce mistakes. Adam is a solid default — adaptive, fast, forgiving of hyperparameter choices.
- **Loss `binary_crossentropy`** = the specific score that measures how wrong the current predictions are. For 0/1 classification, this is the standard choice. Training = minimize this number.
- **Metrics** = extra scores to report as training progresses:
  - `accuracy` — fraction of test cases got right.
  - `AUC` (Area Under the ROC Curve) — how well the model separates the two classes across *every* threshold, not just 0.5. AUC is generally considered a fairer overall score than accuracy alone.

### Step 10 — Train the model (lines 63–65)

```python
print("Training the Deep Neural Network... Please wait.")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
```

- **`epochs=50`** = go through the training data 50 times. Each pass, the model tweaks its weights a little bit to make fewer mistakes.
- **`batch_size=32`** = update the weights after seeing every 32 recordings, not after each one. Faster and more stable than updating one-at-a-time.
- **`verbose=0`** = don't print a progress bar for each epoch. (Change to `verbose=1` if you want to watch training live.)

This is where the "learning" actually happens. All the earlier steps were prep.

### Step 11 — Grade the model (lines 67–72)

```python
loss, accuracy, auc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
```

Show the trained model the 20% it has never seen and see how it does. Two numbers get printed:

- **Accuracy** — what fraction of test recordings were labeled correctly. Target > 70% (the original TIS paper's Random Forest hit ~71%, so the ANN needs to at least match that).
- **AUC** — how well the model *ranks* trustworthy voices above neutral ones. Target > 0.78. AUC doesn't depend on where you draw the 0.5 threshold, so it's a more robust quality measure.

---

## What's NOT in this script (and why)

- **No multi-seed loop.** This script trains once and reports one accuracy/AUC number. If you run it twice you'll get slightly different results because neural network training has random components (weight initialization, dropout). The main experiment in `Analysis/compare_all_models.py` retrains this same architecture 25 times to measure that variability properly.
- **No per-ethnicity fairness gap.** This script only reports overall accuracy. The fairness analysis (which group did well vs poorly) also lives in `compare_all_models.py`.
- **No adversarial fairness.** That's what DANN-Trust adds on top of this base ANN — see `NewMethods/DANN_Trustworthy_Intent_Project/dann_trust_model.py`.

Think of this file as the **educational sample** — a clean, minimal ANN you can read top-to-bottom in five minutes to understand the pattern. The heavy scientific machinery is in `MySolution/Analysis/`.

---

## In one sentence

**`ann_trust_model.py` builds a small three-hidden-layer neural network that takes 60 acoustic voice measurements and predicts whether the speaker was trying to sound trustworthy, using standard tricks (batch normalization, dropout, stratified train/test split, feature scaling) to give the model a fair chance to learn without memorizing.**
