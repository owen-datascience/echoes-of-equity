
# Project: Domain-Adversarial Neural Network (DANN-Trust) for Trustworthy Intent Classification
#
# This script trains an ADVERSARIAL deep learning model that learns to:
#   (1) classify Trustworthy vs. Neutral speech intent (the "good" job), AND
#   (2) FORGET the speaker's ethnicity (the "fair" job).
#
# The trick is the Gradient Reversal Layer (GRL): during training, the
# ethnicity head learns to predict ethnicity normally, BUT the gradient
# that flows back to the shared encoder is MULTIPLIED BY -lambda.  So the
# encoder is pushed to produce features that BREAK the ethnicity head while
# still helping the trust head.  At the end, the encoder ideally produces a
# representation that is informative about trust but uninformative about
# ethnicity.  Reference: Ganin et al., "Domain-Adversarial Training of Neural
# Networks", JMLR 2016.
#
# Expected behaviour:
#   * Trust accuracy   : >70% (matches/beats ANN, CNN baselines)
#   * Per-ethnicity gap: SHRUNK vs. the 5pp gap of the source paper's RF
#   * Probe accuracy   : NEAR 33% (= chance, since there are 3 ethnicities).
#                        High probe accuracy means the encoder leaked ethnicity.

# 1. IMPORTS ------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. GRADIENT REVERSAL LAYER --------------------------------------------------
# Forward pass : identity (just passes the activations through).
# Backward pass: multiplies the upstream gradient by -lambda.
# This is what makes the encoder "fight" the demographic head.
@tf.custom_gradient
def _gradient_reversal_op(x, lambda_):
    def grad(dy):
        # Flip sign and scale by lambda. The second return value (None) is the
        # gradient w.r.t. lambda_ itself, which we never differentiate through.
        return -lambda_ * dy, None
    return tf.identity(x), grad


class GradientReversalLayer(layers.Layer):
    """A Keras layer wrapping the custom-gradient op above.

    Lambda is a non-trainable tf.Variable so we can update it from a callback
    every epoch (Ganin et al.'s schedule).
    """

    def __init__(self, initial_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = tf.Variable(
            float(initial_lambda),
            trainable=False,
            dtype=tf.float32,
            name="grl_lambda",
        )

    def call(self, x):
        return _gradient_reversal_op(x, self.lambda_)

    def set_lambda(self, value):
        self.lambda_.assign(float(value))


# 3. LAMBDA SCHEDULE CALLBACK -------------------------------------------------
# Ganin's recommended schedule:  lambda(p) = 2 / (1 + exp(-10 * p)) - 1
# where p = current_epoch / total_epochs (training-progress fraction).
# Starts near 0 (let the encoder first learn USEFUL features), grows to 1
# (then PUSH it toward demographic invariance).
class LambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, total_epochs, lambda_max=1.0):
        super().__init__()
        self.grl_layer = grl_layer
        self.total_epochs = max(1, int(total_epochs))
        self.lambda_max = float(lambda_max)

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.total_epochs
        # Ganin sigmoid ramps from 0 -> 1; we scale by lambda_max for stronger
        # adversarial pressure (lambda_max > 1 pushes the encoder harder to
        # forget demographics, at some cost to trust accuracy).
        new_lambda = self.lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        self.grl_layer.set_lambda(new_lambda)


# 4. DATA LOADING -------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Speech_dataset_characteristics.csv")
df = pd.read_csv(csv_path)

# Encode the two labels:
#   y_trust  : 0 = Neutral, 1 = Trustworthy  (main task)
#   y_ethn   : 0..K-1 across ethnicities      (adversarial task)
trust_encoder = LabelEncoder()
ethn_encoder = LabelEncoder()
df["Speaker_Intent_enc"] = trust_encoder.fit_transform(df["Speaker_Intent"])
df["Speaker_Ethnicity_enc"] = ethn_encoder.fit_transform(df["Speaker_Ethnicity"])
NUM_ETHNICITIES = int(df["Speaker_Ethnicity_enc"].nunique())
print(f"Loaded {len(df)} utterances.")
print(f"Trust classes      : {list(trust_encoder.classes_)}")
print(f"Ethnicity classes  : {list(ethn_encoder.classes_)} (K = {NUM_ETHNICITIES})")

# Acoustic features only. Drop identifiers / demographics / target columns.
META_COLS = [
    "Audio_Filename", "Speaker_ID", "Speaker_Ethnicity", "Speaker_AgeGroup",
    "Speaker_Sex", "Speaker_Intent", "Sentence_Num",
    "Speaker_Intent_enc", "Speaker_Ethnicity_enc",
]
X = df.drop(columns=META_COLS)
y_trust = df["Speaker_Intent_enc"].values.astype(np.float32)
y_ethn = df["Speaker_Ethnicity_enc"].values.astype(np.int32)
ethn_str = df["Speaker_Ethnicity"].values  # kept for per-group reporting

# Replace any inf, fill NaN with column mean (matches the ANN / CNN scripts).
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
INPUT_DIM = X.shape[1]
print(f"Input feature dim  : {INPUT_DIM}")

# 5. TRAIN / TEST SPLIT -------------------------------------------------------
# Joint stratification on (intent, ethnicity) so each of the 6 cells
# (Neutral/Trust x White/Black/SouthAsian) keeps its proportion in train AND
# test.  Without this, the test set's ethnicity mix can drift away from the
# train set's, which inflates per-ethnicity accuracy variance and biases the
# fairness gap upward.
strata = (y_trust.astype(int) * 10 + y_ethn).astype(int)
(
    X_train, X_test,
    y_trust_train, y_trust_test,
    y_ethn_train, y_ethn_test,
    ethn_str_train, ethn_str_test,
) = train_test_split(
    X.values, y_trust, y_ethn, ethn_str,
    test_size=0.2, stratify=strata, random_state=SEED,
)

# Sanity print so the reader can see the split is balanced.
import collections
print(
    "Test set per-ethnicity counts (Neutral / Trust):",
    {
        e: (
            int(((ethn_str_test == e) & (y_trust_test == 0)).sum()),
            int(((ethn_str_test == e) & (y_trust_test == 1)).sum()),
        )
        for e in ethn_encoder.classes_
    },
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# 6. MODEL ARCHITECTURE -------------------------------------------------------
# Shared encoder G_f  : Dense(128) -> BN -> Drop -> Dense(32) -> BN
# Trust head     G_y  : Dense(16, ReLU) -> Drop -> Dense(1, sigmoid)
# Domain head    G_d  : GRL -> Dense(16, ReLU) -> Drop -> Dense(K, softmax)
def build_dann_model(input_dim, num_domains):
    inputs = layers.Input(shape=(input_dim,), name="acoustic_features")

    # Encoder G_f
    h = layers.Dense(128, activation="relu", name="enc_dense1")(inputs)
    h = layers.BatchNormalization(name="enc_bn1")(h)
    h = layers.Dropout(0.3, name="enc_drop1")(h)
    h = layers.Dense(32, activation="relu", name="enc_dense2")(h)
    encoder_output = layers.BatchNormalization(name="encoder_output")(h)

    # Trust head G_y
    t = layers.Dense(16, activation="relu", name="trust_dense")(encoder_output)
    t = layers.Dropout(0.2, name="trust_drop")(t)
    trust_out = layers.Dense(1, activation="sigmoid", name="trust_output")(t)

    # Domain head G_d (with gradient reversal upstream)
    grl = GradientReversalLayer(initial_lambda=0.0, name="grl")
    d = grl(encoder_output)
    d = layers.Dense(16, activation="relu", name="dom_dense")(d)
    d = layers.Dropout(0.2, name="dom_drop")(d)
    domain_out = layers.Dense(num_domains, activation="softmax", name="domain_output")(d)

    model = Model(inputs=inputs, outputs=[trust_out, domain_out], name="DANN_Trust")
    # Stash the GRL layer for the callback.
    model.grl_layer = grl
    # Build an encoder-only model so we can extract features for the probe test.
    encoder_only = Model(inputs=inputs, outputs=encoder_output, name="encoder")
    model.encoder_only = encoder_only
    return model


model = build_dann_model(INPUT_DIM, NUM_ETHNICITIES)
model.summary()

# 7. COMPILE ------------------------------------------------------------------
# Two losses, both with weight 1.0. The GRL is what makes the domain loss
# adversarial w.r.t. the encoder; from Keras's view, both are normal losses.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "trust_output": "binary_crossentropy",
        "domain_output": "sparse_categorical_crossentropy",
    },
    loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    metrics={
        "trust_output": [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
        "domain_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    },
)

# 8. TRAIN --------------------------------------------------------------------
TOTAL_EPOCHS = 80
LAMBDA_MAX = 1.0    # Adversarial strength; 1.0 = Ganin's default.
print(f"\nTraining DANN-Trust for {TOTAL_EPOCHS} epochs, lambda_max={LAMBDA_MAX}...")
history = model.fit(
    X_train_scaled,
    {"trust_output": y_trust_train, "domain_output": y_ethn_train},
    epochs=TOTAL_EPOCHS,
    batch_size=32,
    verbose=0,
    callbacks=[LambdaScheduler(model.grl_layer, TOTAL_EPOCHS, lambda_max=LAMBDA_MAX)],
)
print(f"Final lambda value : {float(model.grl_layer.lambda_.numpy()):.3f}")

# 9. EVALUATE -----------------------------------------------------------------
print("\n=== OVERALL EVALUATION ===")
trust_probs, _ = model.predict(X_test_scaled, verbose=0)
trust_probs = trust_probs.ravel()
trust_preds = (trust_probs >= 0.5).astype(np.int32)

acc = accuracy_score(y_trust_test, trust_preds)
auc = roc_auc_score(y_trust_test, trust_probs)
print(f"Trust Accuracy     : {acc * 100:.2f}%  (Target: >70%)")
print(f"Trust AUC          : {auc:.3f}        (Target: >0.78)")
print("\nClassification report (trust head):")
print(classification_report(
    y_trust_test, trust_preds,
    target_names=list(trust_encoder.classes_),
    digits=3,
))

# 10. PER-ETHNICITY FAIRNESS METRICS -----------------------------------------
print("\n=== PER-ETHNICITY FAIRNESS ===")
per_ethn_acc = {}
per_ethn_auc = {}
for ethn_name in ethn_encoder.classes_:
    mask = (ethn_str_test == ethn_name)
    if mask.sum() < 2:
        continue
    g_acc = accuracy_score(y_trust_test[mask], trust_preds[mask])
    # AUC needs both classes present in the slice.
    try:
        g_auc = roc_auc_score(y_trust_test[mask], trust_probs[mask])
    except ValueError:
        g_auc = float("nan")
    per_ethn_acc[ethn_name] = g_acc
    per_ethn_auc[ethn_name] = g_auc
    print(f"  {ethn_name:<12} N = {int(mask.sum()):>3}  "
          f"acc = {g_acc * 100:5.2f}%   auc = {g_auc:.3f}")

if per_ethn_acc:
    gap_pp = (max(per_ethn_acc.values()) - min(per_ethn_acc.values())) * 100
    print(f"\nFairness gap (max - min ethnicity accuracy): {gap_pp:.2f}pp")
    print(f"  Source paper RF gap was 5pp (White 71%, South Asian 66%).")

# 11. PROBE CLASSIFIER -------------------------------------------------------
# Train a simple LR on the FROZEN encoder features to predict ethnicity.
# If the adversary worked, this should be close to 1/K = 33%.  If it's still
# high (say >50%), the encoder is leaking ethnicity and lambda should grow.
print("\n=== PROBE CLASSIFIER (lower = better debiasing) ===")
enc_train = model.encoder_only.predict(X_train_scaled, verbose=0)
enc_test = model.encoder_only.predict(X_test_scaled, verbose=0)

probe = LogisticRegression(max_iter=2000, random_state=SEED)
probe.fit(enc_train, y_ethn_train)
probe_acc = accuracy_score(y_ethn_test, probe.predict(enc_test))
chance = 1.0 / NUM_ETHNICITIES
print(f"Probe ethnicity accuracy from encoder : {probe_acc * 100:.2f}%")
print(f"Chance level (1 / K)                  : {chance * 100:.2f}%")
print(f"Leakage above chance                  : {(probe_acc - chance) * 100:+.2f}pp")

print("\nDone.  If trust acc > 70% AND probe acc is close to chance,")
print("DANN-Trust has successfully decoupled trust prediction from ethnicity.")
