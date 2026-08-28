
# Echoes of Equity: DANN-Trust ablation study (Section 6.7, Table 8).
#
# Runs DANN with one component removed per configuration, on the SAME
# fixed 80/20 split and the SAME 5 seeds as compare_all_models.py, so
# every row of Table 8 is directly comparable to the full-DANN row of
# Table 3.
#
# Variants:
#   dann_full       : reference (identical to run_dann in the main harness)
#   dann_no_grl     : GRL replaced with tf.identity => multi-task learner
#   dann_no_domain  : domain head deleted => ANN with 2-layer encoder (128 -> 32)
#   dann_no_bn      : BatchNormalization removed from encoder
#   dann_top15      : input restricted to the 15 highest-Gini RF features
#
# Output: ablation_results.json (mean +/- std of Acc, AUC and Δeth per variant).
# The "top-15 features only" variant reads RF feature importances from the
# main results.json produced by compare_all_models.py; run that first.

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# 50 seeds to match compare_all_models.py so Table 8 rows are directly
# comparable to Table 3 rows.  Set back to list(range(42, 67)) or
# list(range(42, 47)) for smaller runs.
SEEDS = list(range(42, 92))
SPLIT_SEED = 42
EPOCHS = 80
LAM_MAX = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "NewMethods", "DANN_Trustworthy_Intent_Project",
    "Speech_dataset_characteristics.csv",
))
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results.json")
OUT_JSON = os.path.join(SCRIPT_DIR, "ablation_results.json")


# ----------------------------------------------------------------------------
# Gradient Reversal Layer (same as compare_all_models.py -- duplicated to keep
# this script runnable without triggering the full 5-model harness on import).
# ----------------------------------------------------------------------------
@tf.custom_gradient
def _grl_op(x, lam):
    def grad(dy):
        return -lam * dy, None
    return tf.identity(x), grad


class GRL(layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.lam = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="grl_lambda")

    def call(self, x):
        return _grl_op(x, self.lam)

    def set_lambda(self, v):
        self.lam.assign(float(v))


class LambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, grl, epochs, lam_max=1.0):
        super().__init__()
        self.grl = grl
        self.epochs = max(1, int(epochs))
        self.lam_max = float(lam_max)

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.epochs
        self.grl.set_lambda(self.lam_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


# ----------------------------------------------------------------------------
# Shared data loading (mirror of compare_all_models.py so this script stands
# alone; importing that module would run the full 5-model comparison).
# ----------------------------------------------------------------------------
print(f"Loading {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

trust_enc = LabelEncoder()
ethn_enc = LabelEncoder()
df["y_trust"] = trust_enc.fit_transform(df["Speaker_Intent"])
df["y_ethn"] = ethn_enc.fit_transform(df["Speaker_Ethnicity"])

META_COLS = [
    "Audio_Filename", "Speaker_ID", "Speaker_Ethnicity", "Speaker_AgeGroup",
    "Speaker_Sex", "Speaker_Intent", "Sentence_Num", "y_trust", "y_ethn",
]
X_df = df.drop(columns=META_COLS)
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(X_df.mean())

X = X_df.values.astype(np.float32)
FEATURE_NAMES = X_df.columns.tolist()
y_trust = df["y_trust"].values.astype(np.float32)
y_ethn = df["y_ethn"].values.astype(np.int32)
ethn_str = df["Speaker_Ethnicity"].values

strata = (y_trust.astype(int) * 10 + y_ethn).astype(int)
X_tr, X_te, yt_tr, yt_te, ye_tr, ye_te, es_tr, es_te = train_test_split(
    X, y_trust, y_ethn, ethn_str,
    test_size=0.2, stratify=strata, random_state=SPLIT_SEED,
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

INPUT_DIM = X_tr_s.shape[1]
NUM_ETH = int(df["y_ethn"].nunique())


def _seed_all(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _eval(probs):
    preds = (probs >= 0.5).astype(np.int32)
    accs = []
    for e in ethn_enc.classes_:
        mask = (es_te == e)
        if mask.sum() < 2:
            continue
        accs.append(accuracy_score(yt_te[mask], preds[mask]))
    return {
        "acc": float(accuracy_score(yt_te, preds)),
        "auc": float(roc_auc_score(yt_te, probs)),
        "gap_pp": (max(accs) - min(accs)) * 100 if accs else 0.0,
    }


# ----------------------------------------------------------------------------
# Encoder builders.  Kept small and explicit so each ablation reads at a glance.
# ----------------------------------------------------------------------------
def _encoder(inp, use_bn=True):
    """DANN encoder: Dense 128 -> [BN] -> Dropout -> Dense 32 -> [BN]."""
    h = Dense(128, activation="relu")(inp)
    if use_bn:
        h = BatchNormalization()(h)
    h = Dropout(0.3)(h)
    h = Dense(32, activation="relu")(h)
    if use_bn:
        h = BatchNormalization(name="encoder_output")(h)
    return h


def _trust_head(enc):
    t = Dense(16, activation="relu")(enc)
    t = Dropout(0.2)(t)
    return Dense(1, activation="sigmoid", name="trust_output")(t)


def _domain_head(enc, num_classes):
    d = Dense(16, activation="relu")(enc)
    d = Dropout(0.2)(d)
    return Dense(num_classes, activation="softmax", name="domain_output")(d)


# ----------------------------------------------------------------------------
# Ablation variants.  Each returns test-set probabilities on the trust class.
# ----------------------------------------------------------------------------
def run_dann_full(seed, X_tr_v=None, X_te_v=None, input_dim=None):
    _seed_all(seed)
    input_dim = input_dim or INPUT_DIM
    X_tr_v = X_tr_s if X_tr_v is None else X_tr_v
    X_te_v = X_te_s if X_te_v is None else X_te_v
    inp = Input(shape=(input_dim,))
    enc = _encoder(inp, use_bn=True)
    trust_out = _trust_head(enc)
    grl = GRL(name="grl")
    dom_out = _domain_head(grl(enc), NUM_ETH)
    m = Model(inp, [trust_out, dom_out])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"trust_output": "binary_crossentropy",
              "domain_output": "sparse_categorical_crossentropy"},
        loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    )
    m.fit(X_tr_v,
          {"trust_output": yt_tr, "domain_output": ye_tr},
          epochs=EPOCHS, batch_size=32, verbose=0,
          callbacks=[LambdaScheduler(grl, EPOCHS, LAM_MAX)])
    return m.predict(X_te_v, verbose=0)[0].ravel()


def run_dann_no_grl(seed):
    """Domain head still trained, but no gradient reversal -- pure multi-task."""
    _seed_all(seed)
    inp = Input(shape=(INPUT_DIM,))
    enc = _encoder(inp, use_bn=True)
    trust_out = _trust_head(enc)
    dom_out = _domain_head(enc, NUM_ETH)   # <-- direct connection, no GRL
    m = Model(inp, [trust_out, dom_out])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"trust_output": "binary_crossentropy",
              "domain_output": "sparse_categorical_crossentropy"},
        loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    )
    m.fit(X_tr_s,
          {"trust_output": yt_tr, "domain_output": ye_tr},
          epochs=EPOCHS, batch_size=32, verbose=0)
    return m.predict(X_te_s, verbose=0)[0].ravel()


def run_dann_no_domain(seed):
    """Encoder + trust head only -- the ANN with a 2-layer (128->32) encoder."""
    _seed_all(seed)
    inp = Input(shape=(INPUT_DIM,))
    enc = _encoder(inp, use_bn=True)
    trust_out = _trust_head(enc)
    m = Model(inp, trust_out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(X_tr_s, yt_tr, epochs=EPOCHS, batch_size=32, verbose=0)
    return m.predict(X_te_s, verbose=0).ravel()


def run_dann_no_bn(seed):
    _seed_all(seed)
    inp = Input(shape=(INPUT_DIM,))
    enc = _encoder(inp, use_bn=False)
    trust_out = _trust_head(enc)
    grl = GRL(name="grl")
    dom_out = _domain_head(grl(enc), NUM_ETH)
    m = Model(inp, [trust_out, dom_out])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"trust_output": "binary_crossentropy",
              "domain_output": "sparse_categorical_crossentropy"},
        loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    )
    m.fit(X_tr_s,
          {"trust_output": yt_tr, "domain_output": ye_tr},
          epochs=EPOCHS, batch_size=32, verbose=0,
          callbacks=[LambdaScheduler(grl, EPOCHS, LAM_MAX)])
    return m.predict(X_te_s, verbose=0)[0].ravel()


def _top_k_feature_indices(k=15):
    """Read RF feature importances (mean across seeds) from results.json."""
    if not os.path.exists(RESULTS_JSON):
        raise SystemExit(
            f"{RESULTS_JSON} not found -- run compare_all_models.py first so "
            "the top-15 ablation has RF feature importances to pick from."
        )
    with open(RESULTS_JSON) as f:
        res = json.load(f)
    fi = res.get("RF", {}).get("feature_importances_mean")
    if fi is None:
        raise SystemExit(
            "RF feature_importances_mean missing from results.json.  "
            "Re-run compare_all_models.py (the current head writes it)."
        )
    return np.argsort(fi)[::-1][:k]


def run_dann_top15(seed, top_idx):
    X_tr_t = X_tr_s[:, top_idx]
    X_te_t = X_te_s[:, top_idx]
    return run_dann_full(seed, X_tr_v=X_tr_t, X_te_v=X_te_t, input_dim=len(top_idx))


# ----------------------------------------------------------------------------
# Run every variant across every seed.
# ----------------------------------------------------------------------------
top_idx = _top_k_feature_indices(15)
top_names = [FEATURE_NAMES[i] for i in top_idx]
print(f"Top-15 features (mean RF Gini): {top_names}")

VARIANTS = [
    ("dann_full",      lambda s: run_dann_full(s)),
    ("dann_no_grl",    run_dann_no_grl),
    ("dann_no_domain", run_dann_no_domain),
    ("dann_no_bn",     run_dann_no_bn),
    ("dann_top15",     lambda s: run_dann_top15(s, top_idx)),
]

ablation = {}
# Resume support: if a prior run finished some variants (with the SAME SEEDS
# list), keep them and skip. Delete ablation_results.json to force a full
# rerun. A stale cache with a different seed count is detected and rejected.
if os.path.exists(OUT_JSON):
    try:
        with open(OUT_JSON) as f:
            prior = json.load(f)
        # Only reuse entries whose recorded seed list matches ours exactly.
        for k, v in prior.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and v.get("seeds") == SEEDS:
                ablation[k] = v
        if ablation:
            print(f"Loaded {len(ablation)} previously completed variants "
                  f"(matching {len(SEEDS)}-seed run); will skip.")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read {OUT_JSON} ({e}); starting fresh.")

for name, fn in VARIANTS:
    if name in ablation:
        print(f"[skip] {name}: already in {OUT_JSON}")
        continue
    print(f"\n=== {name}: running {len(SEEDS)} seeds ===", flush=True)
    per_seed = []
    for s in SEEDS:
        probs = fn(s)
        r = _eval(probs)
        per_seed.append(r)
        print(f"  seed={s}  acc={r['acc']*100:5.2f}%  auc={r['auc']:.3f}  gap={r['gap_pp']:5.2f}pp",
              flush=True)
    accs = [r["acc"] for r in per_seed]
    aucs = [r["auc"] for r in per_seed]
    gaps = [r["gap_pp"] for r in per_seed]
    ablation[name] = {
        "seeds": SEEDS,
        "acc_mean": float(np.mean(accs)),  "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),  "auc_std": float(np.std(aucs)),
        "gap_pp_mean": float(np.mean(gaps)), "gap_pp_std": float(np.std(gaps)),
        "raw_per_seed": per_seed,
    }
    # Persist after every variant so a crash/interrupt doesn't lose progress.
    ablation["_top15_feature_indices"] = [int(i) for i in top_idx]
    ablation["_top15_feature_names"] = top_names
    with open(OUT_JSON, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"[saved] {OUT_JSON}")

# ----------------------------------------------------------------------------
# Table 8 (mean +/- std) to stdout.
# ----------------------------------------------------------------------------
print("\n" + "=" * 88)
print(f"TABLE 8: DANN-Trust ablation  (mean +/- std over {len(SEEDS)} seeds; split seed=42)")
print("=" * 88)
print(f"{'Variant':<18} {'Accuracy':>16} {'AUC':>14} {'Δeth (pp)':>16}")
print("-" * 88)
for name, r in ablation.items():
    if name.startswith("_"):
        continue
    print(
        f"{name:<18} "
        f"{r['acc_mean']*100:>7.2f}% +/-{r['acc_std']*100:4.2f}  "
        f"{r['auc_mean']:>6.3f} +/-{r['auc_std']:.3f}  "
        f"{r['gap_pp_mean']:>8.2f}pp +/-{r['gap_pp_std']:4.2f}"
    )

# ----------------------------------------------------------------------------
# Levene's test: is dann_full's fairness-gap VARIANCE actually different from
# each ablation variant? p<0.05 => yes.  scipy ships as a scikit-learn dep,
# so it's already available.
# ----------------------------------------------------------------------------
try:
    from scipy import stats
except ImportError:
    print("\n[warn] scipy not installed; skipping Levene's test.")
else:
    def _gaps(name):
        return [r["gap_pp"] for r in ablation[name]["raw_per_seed"]]

    pairs = [
        ("dann_full", "dann_no_grl"),
        ("dann_full", "dann_no_domain"),
        ("dann_full", "dann_no_bn"),
        ("dann_full", "dann_top15"),
    ]

    # ----------------------------------------------------------------------
    # Mann-Whitney U on GAP MEAN and ACCURACY MEAN.  The ablation's most
    # important finding (dann_no_bn dominates dann_full on both metrics)
    # only becomes a paper-defensible claim once MEANS are statistically
    # tested; Levene's below only covers VARIANCE of the fairness gap.
    # ----------------------------------------------------------------------
    def _accs(name):
        return [r["acc"] for r in ablation[name]["raw_per_seed"]]

    print("\n" + "=" * 72)
    print(f"GAP MEAN comparison (Mann-Whitney U, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<32} {'gap_a':>7} {'gap_b':>7} {'U':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in pairs:
        if a not in ablation or b not in ablation:
            continue
        ga, gb = _gaps(a), _gaps(b)
        mu_a, mu_b = float(np.mean(ga)), float(np.mean(gb))
        u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<20} {mu_a:>7.2f} {mu_b:>7.2f} {u:>8.1f} {p:>8.4f}   {tag}")

    print("\n" + "=" * 72)
    print(f"ACCURACY MEAN comparison (Mann-Whitney U, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<32} {'acc_a%':>7} {'acc_b%':>7} {'U':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in pairs:
        if a not in ablation or b not in ablation:
            continue
        aa, ab = _accs(a), _accs(b)
        mu_a, mu_b = float(np.mean(aa)) * 100, float(np.mean(ab)) * 100
        u, p = stats.mannwhitneyu(aa, ab, alternative="two-sided")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<20} {mu_a:>7.2f} {mu_b:>7.2f} {u:>8.1f} {p:>8.4f}   {tag}")

    print("\n" + "=" * 72)
    print(f"GAP VARIANCE comparison (Levene's test, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<40} {'W':>7} {'p-value':>10}   {'verdict'}")
    print("-" * 72)
    for a, b in pairs:
        if a not in ablation or b not in ablation:
            continue
        stat, p = stats.levene(_gaps(a), _gaps(b), center="median")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<22} {stat:>7.2f} {p:>10.4f}   {tag}")

print(f"\nSaved ablation results to {OUT_JSON}")
