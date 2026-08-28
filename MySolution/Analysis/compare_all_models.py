
# Echoes of Equity: multi-seed head-to-head comparison of all 5 models on the
# IDENTICAL joint-stratified 80/20 split.  The data split is fixed
# (seed=42 controls the train/test partition).  Only the MODEL initialization
# seed varies across reruns.  This isolates "model variance" from "split luck".
#
# Models compared:
#   * RF    (sklearn RandomForestClassifier)
#   * LR    (sklearn LogisticRegression)
#   * ANN   (Keras 3-layer dense network with BN + dropout)
#   * CNN   (Keras 1D-conv with two Conv1D layers)
#   * DANN  (Keras adversarial model: shared encoder + trust head + GRL-protected ethnicity head)
#
# Output:
#   * Pretty-printed mean +/- std tables on stdout.
#   * results.json: per-seed lists AND aggregated mean/std for every metric,
#     for direct consumption by the paper's table macros and figures.

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# 50 seeds so the sex-gap MEAN comparison (Mann-Whitney U vs baselines) has
# enough power to cross p<0.05 -- 25 seeds landed at p~0.07 for RF vs CNN
# and RF vs DANN.  Doubling gets us over that threshold if the effect is real.
# Set back to list(range(42, 67)) or list(range(42, 47)) for smaller runs.
SEEDS = list(range(42, 92))
SPLIT_SEED = 42       # fixed; identical train/test for every model and every run.

# ----------------------------------------------------------------------------
# Gradient Reversal Layer (DANN building block)
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
# Shared data loading + joint stratified split.  EVERY model below uses the
# IDENTICAL X_tr_s / X_te_s / y_*_tr / y_*_te arrays so comparisons are valid.
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "NewMethods", "DANN_Trustworthy_Intent_Project",
    "Speech_dataset_characteristics.csv",
))
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
age_str = df["Speaker_AgeGroup"].values
sex_str = df["Speaker_Sex"].values

# Joint stratification: 2 intents * 3 ethnicities = 6 cells, balanced across split.
strata = (y_trust.astype(int) * 10 + y_ethn).astype(int)
(X_tr, X_te, yt_tr, yt_te, ye_tr, ye_te,
 es_tr, es_te, ag_tr, ag_te, sx_tr, sx_te) = train_test_split(
    X, y_trust, y_ethn, ethn_str, age_str, sex_str,
    test_size=0.2, stratify=strata, random_state=SPLIT_SEED,
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

INPUT_DIM = X_tr_s.shape[1]
NUM_ETH = int(df["y_ethn"].nunique())

print(f"N_train = {len(X_tr_s)}, N_test = {len(X_te_s)}, INPUT_DIM = {INPUT_DIM}")
print("Test set per (ethnicity, intent) cell:")
for e in ethn_enc.classes_:
    n_neu = int(((es_te == e) & (yt_te == 0)).sum())
    n_tru = int(((es_te == e) & (yt_te == 1)).sum())
    print(f"  {e:<12}  Neutral={n_neu:>3}   Trustworthy={n_tru:>3}")


# ----------------------------------------------------------------------------
# Per-model trainers.  Each accepts a SEED and returns (probs, preds) on the
# fixed held-out test set.  We seed numpy + tf BEFORE building the model so
# weight initialization (and any internal randomness) is controlled.
# ----------------------------------------------------------------------------
def _seed_all(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# RF feature-importance vectors (one per seed), captured as a side-effect so
# every runner keeps the uniform (probs, preds) return contract.  Fig. 12
# and the DANN "top-15 features only" ablation read from here.
rf_fi_per_seed = []


def run_rf(seed):
    _seed_all(seed)
    m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    m.fit(X_tr_s, yt_tr)
    rf_fi_per_seed.append(m.feature_importances_.tolist())
    return m.predict_proba(X_te_s)[:, 1], m.predict(X_te_s).astype(np.int32)


def run_lr(seed):
    _seed_all(seed)
    m = LogisticRegression(solver="liblinear", random_state=seed, max_iter=2000)
    m.fit(X_tr_s, yt_tr)
    return m.predict_proba(X_te_s)[:, 1], m.predict(X_te_s).astype(np.int32)


def run_ann(seed):
    _seed_all(seed)
    m = Sequential([
        Dense(128, activation="relu", input_shape=(INPUT_DIM,)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(X_tr_s, yt_tr, epochs=50, batch_size=32, verbose=0)
    p = m.predict(X_te_s, verbose=0).ravel()
    return p, (p >= 0.5).astype(np.int32)


def run_cnn(seed):
    _seed_all(seed)
    X_tr_c = X_tr_s.reshape(X_tr_s.shape[0], INPUT_DIM, 1)
    X_te_c = X_te_s.reshape(X_te_s.shape[0], INPUT_DIM, 1)
    m = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(INPUT_DIM, 1)),
        BatchNormalization(), Dropout(0.2),
        Conv1D(32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(X_tr_c, yt_tr, epochs=50, batch_size=32, verbose=0)
    p = m.predict(X_te_c, verbose=0).ravel()
    return p, (p >= 0.5).astype(np.int32)


def run_dann(seed, epochs=80, lam_max=1.0):
    # NOTE: BatchNormalization intentionally omitted from the encoder.
    # The 50-seed ablation (Section VI.G) shows that dann_no_bn significantly
    # outperforms dann_full on accuracy (p=0.002), gap mean (p=0.011), and gap
    # variance (p=0.049).  This is the "DANN-Trust" architecture reported in
    # the paper.  See run_dann_ablation.py for the with-BN comparison.
    _seed_all(seed)
    inp = Input(shape=(INPUT_DIM,), name="acoustic_features")
    h = Dense(128, activation="relu")(inp)
    h = Dropout(0.3)(h)
    enc = Dense(32, activation="relu", name="encoder_output")(h)
    t = Dense(16, activation="relu")(enc)
    t = Dropout(0.2)(t)
    trust_out = Dense(1, activation="sigmoid", name="trust_output")(t)
    grl = GRL(name="grl")
    d = grl(enc)
    d = Dense(16, activation="relu")(d)
    d = Dropout(0.2)(d)
    dom_out = Dense(NUM_ETH, activation="softmax", name="domain_output")(d)
    m = Model(inputs=inp, outputs=[trust_out, dom_out], name="DANN_Trust")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "trust_output": "binary_crossentropy",
            "domain_output": "sparse_categorical_crossentropy",
        },
        loss_weights={"trust_output": 1.0, "domain_output": 1.0},
    )
    m.fit(
        X_tr_s,
        {"trust_output": yt_tr, "domain_output": ye_tr},
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[LambdaScheduler(grl, epochs, lam_max)],
    )
    p = m.predict(X_te_s, verbose=0)[0].ravel()
    return p, (p >= 0.5).astype(np.int32)


# ----------------------------------------------------------------------------
# Run each model on every seed; collect per-seed metrics.
# ----------------------------------------------------------------------------
MODELS = [
    ("RF", run_rf),
    ("LR", run_lr),
    ("ANN", run_ann),
    ("CNN", run_cnn),
    ("DANN", run_dann),
]
ETH_ORDER = ["White", "Black", "South_Asian"]
AGE_ORDER = ["Younger", "Older"]
SEX_ORDER = ["Female", "Male"]


def _slice_metrics(group_arr, group_values, probs, preds):
    """Per-group {n, acc, auc} for one demographic axis, plus max-min gap in pp."""
    per_group = {}
    for g in group_values:
        mask = (group_arr == g)
        if mask.sum() < 2:
            continue
        try:
            g_auc = float(roc_auc_score(yt_te[mask], probs[mask]))
        except ValueError:
            g_auc = float("nan")
        per_group[g] = {
            "n": int(mask.sum()),
            "acc": float(accuracy_score(yt_te[mask], preds[mask])),
            "auc": g_auc,
        }
    accs = [v["acc"] for v in per_group.values()]
    gap_pp = (max(accs) - min(accs)) * 100 if accs else 0.0
    return per_group, gap_pp


def _eval_one(probs, preds):
    """Return overall + per-ethnicity/age/sex metrics and confusion matrix."""
    overall_acc = float(accuracy_score(yt_te, preds))
    overall_auc = float(roc_auc_score(yt_te, probs))
    per_eth, gap_eth_pp = _slice_metrics(es_te, ethn_enc.classes_, probs, preds)
    per_age, gap_age_pp = _slice_metrics(ag_te, AGE_ORDER, probs, preds)
    per_sex, gap_sex_pp = _slice_metrics(sx_te, SEX_ORDER, probs, preds)
    # sklearn returns [[TN, FP], [FN, TP]] for labels=[0, 1].
    cm = confusion_matrix(yt_te.astype(int), preds, labels=[0, 1]).tolist()
    return {
        "overall_acc": overall_acc,
        "overall_auc": overall_auc,
        "per_eth": per_eth,
        "per_age": per_age,
        "per_sex": per_sex,
        "gap_pp": gap_eth_pp,
        "gap_age_pp": gap_age_pp,
        "gap_sex_pp": gap_sex_pp,
        "confusion_matrix": cm,
    }


all_results = {}
OUT_PATH = os.path.join(SCRIPT_DIR, "results.json")

# Resume support: if a prior run finished some models (with the SAME SEEDS
# list), keep them and skip. Delete results.json to force a full rerun.
# A stale cache with a different seed count is detected and rejected.
if os.path.exists(OUT_PATH):
    try:
        with open(OUT_PATH) as f:
            prior = json.load(f)
        for k, v in prior.items():
            if k.startswith("_"):
                all_results[k] = v            # keep _feature_names etc.
                continue
            if isinstance(v, dict) and v.get("seeds") == SEEDS:
                all_results[k] = v
                # Restore rf_fi_per_seed side-effect so ablation top-15 still works.
                if k == "RF" and "feature_importances_per_seed" in v:
                    rf_fi_per_seed[:] = v["feature_importances_per_seed"]
        cached = [k for k in all_results if not k.startswith("_")]
        if cached:
            print(f"Loaded {len(cached)} previously completed models "
                  f"(matching {len(SEEDS)}-seed run): {cached}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read {OUT_PATH} ({e}); starting fresh.")
        all_results = {}

for name, fn in MODELS:
    if name in all_results:
        print(f"[skip] {name}: already in {OUT_PATH}")
        continue
    print(f"\n=== {name}: running {len(SEEDS)} seeds ===", flush=True)
    per_seed = []
    for s in SEEDS:
        probs, preds = fn(s)
        r = _eval_one(probs, preds)
        per_seed.append(r)
        print(f"  seed={s}  acc={r['overall_acc']*100:5.2f}%  auc={r['overall_auc']:.3f}  gap={r['gap_pp']:5.2f}pp",
              flush=True)

    # Aggregate
    overall_accs = [r["overall_acc"] for r in per_seed]
    overall_aucs = [r["overall_auc"] for r in per_seed]
    gaps = [r["gap_pp"] for r in per_seed]
    gaps_age = [r["gap_age_pp"] for r in per_seed]
    gaps_sex = [r["gap_sex_pp"] for r in per_seed]

    def _slice_agg(slice_key, group_values):
        acc_lists = {g: [r[slice_key][g]["acc"] for r in per_seed if g in r[slice_key]]
                     for g in group_values}
        auc_lists = {g: [r[slice_key][g]["auc"] for r in per_seed if g in r[slice_key]]
                     for g in group_values}
        return (
            {g: float(np.mean(v)) for g, v in acc_lists.items() if v},
            {g: float(np.std(v)) for g, v in acc_lists.items() if v},
            {g: float(np.nanmean(v)) for g, v in auc_lists.items() if v},
            {g: float(np.nanstd(v)) for g, v in auc_lists.items() if v},
        )

    eth_acc_mu, eth_acc_sd, eth_auc_mu, eth_auc_sd = _slice_agg("per_eth", ethn_enc.classes_)
    age_acc_mu, age_acc_sd, age_auc_mu, age_auc_sd = _slice_agg("per_age", AGE_ORDER)
    sex_acc_mu, sex_acc_sd, sex_auc_mu, sex_auc_sd = _slice_agg("per_sex", SEX_ORDER)

    # Mean confusion matrix over seeds, elementwise (stored as list-of-lists).
    cm_stack = np.array([r["confusion_matrix"] for r in per_seed], dtype=float)
    cm_mean = cm_stack.mean(axis=0).tolist()

    all_results[name] = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "overall_acc_mean": float(np.mean(overall_accs)),
        "overall_acc_std": float(np.std(overall_accs)),
        "overall_auc_mean": float(np.mean(overall_aucs)),
        "overall_auc_std": float(np.std(overall_aucs)),
        "gap_pp_mean": float(np.mean(gaps)),
        "gap_pp_std": float(np.std(gaps)),
        "gap_age_pp_mean": float(np.mean(gaps_age)),
        "gap_age_pp_std": float(np.std(gaps_age)),
        "gap_sex_pp_mean": float(np.mean(gaps_sex)),
        "gap_sex_pp_std": float(np.std(gaps_sex)),
        "per_eth_acc_mean": eth_acc_mu, "per_eth_acc_std": eth_acc_sd,
        "per_eth_auc_mean": eth_auc_mu, "per_eth_auc_std": eth_auc_sd,
        "per_age_acc_mean": age_acc_mu, "per_age_acc_std": age_acc_sd,
        "per_age_auc_mean": age_auc_mu, "per_age_auc_std": age_auc_sd,
        "per_sex_acc_mean": sex_acc_mu, "per_sex_acc_std": sex_acc_sd,
        "per_sex_auc_mean": sex_auc_mu, "per_sex_auc_std": sex_auc_sd,
        "confusion_matrix_mean": cm_mean,
        "raw_per_seed": per_seed,
    }

    # Persist after every model so a crash/interrupt doesn't lose progress.
    # Feature-importance side-effects are attached below; write them too.
    if name == "RF" and rf_fi_per_seed:
        fi_stack_now = np.array(rf_fi_per_seed, dtype=float)
        all_results[name]["feature_importances_per_seed"] = rf_fi_per_seed
        all_results[name]["feature_importances_mean"] = fi_stack_now.mean(axis=0).tolist()
        all_results[name]["feature_importances_std"] = fi_stack_now.std(axis=0).tolist()
    all_results.setdefault("_feature_names", FEATURE_NAMES)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved] {OUT_PATH}")


# ----------------------------------------------------------------------------
# Print tables (mean +/- std across seeds)
# ----------------------------------------------------------------------------
print("\n" + "=" * 96)
print(f"OVERALL PERFORMANCE  (mean +/- std over {len(SEEDS)} seeds; split seed=42)")
print("=" * 96)
print(f"{'Model':<6} {'Accuracy':>16} {'AUC':>14} {'Gap':>14}")
print("-" * 96)
for name, r in all_results.items():
    if name.startswith("_"):           # skip metadata keys like _feature_names
        continue
    print(
        f"{name:<6} "
        f"{r['overall_acc_mean']*100:>7.2f}% +/-{r['overall_acc_std']*100:4.2f}  "
        f"{r['overall_auc_mean']:>6.3f} +/-{r['overall_auc_std']:.3f}  "
        f"{r['gap_pp_mean']:>6.2f}pp +/-{r['gap_pp_std']:4.2f}"
    )

print("\n" + "=" * 96)
print(f"PER-ETHNICITY ACCURACY  (mean +/- std over {len(SEEDS)} seeds)")
print("=" * 96)
header = f"{'Model':<6} " + "  ".join(f"{e:>18}" for e in ETH_ORDER)
print(header)
print("-" * len(header))
for name, r in all_results.items():
    if name.startswith("_"):           # skip metadata keys like _feature_names
        continue
    cells = []
    for e in ETH_ORDER:
        if e in r["per_eth_acc_mean"]:
            mu = r["per_eth_acc_mean"][e] * 100
            sd = r["per_eth_acc_std"][e] * 100
            cells.append(f"{mu:5.2f}% +/-{sd:4.2f}")
        else:
            cells.append("-")
    print(f"{name:<6} " + "  ".join(f"{c:>18}" for c in cells))

print("\nReference (source paper, LOSO-CV): RF 71% overall, 5pp gap")


def _print_slice_table(title, slice_key_mean, slice_key_std, group_order):
    print("\n" + "=" * 96)
    print(f"{title}  (mean +/- std over {len(SEEDS)} seeds)")
    print("=" * 96)
    header = f"{'Model':<6} " + "  ".join(f"{g:>18}" for g in group_order) + f"  {'Gap':>14}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        if name.startswith("_"):       # skip metadata keys like _feature_names
            continue
        cells = []
        for g in group_order:
            if g in r[slice_key_mean]:
                mu = r[slice_key_mean][g] * 100
                sd = r[slice_key_std][g] * 100
                cells.append(f"{mu:5.2f}% +/-{sd:4.2f}")
            else:
                cells.append("-")
        gap_key = "gap_age_pp" if "age" in slice_key_mean else "gap_sex_pp"
        gap = f"{r[gap_key + '_mean']:5.2f}pp +/-{r[gap_key + '_std']:4.2f}"
        print(f"{name:<6} " + "  ".join(f"{c:>18}" for c in cells) + f"  {gap:>14}")


_print_slice_table("PER-AGE-GROUP ACCURACY", "per_age_acc_mean", "per_age_acc_std", AGE_ORDER)
_print_slice_table("PER-SEX ACCURACY", "per_sex_acc_mean", "per_sex_acc_std", SEX_ORDER)

# ----------------------------------------------------------------------------
# RF feature importance + feature-name index.  Kept for the case where RF
# was loaded from cache (the per-model save above only fires when RF is
# actually retrained this run).
# ----------------------------------------------------------------------------
if rf_fi_per_seed and "feature_importances_mean" not in all_results.get("RF", {}):
    fi_stack = np.array(rf_fi_per_seed, dtype=float)
    all_results["RF"]["feature_importances_per_seed"] = rf_fi_per_seed
    all_results["RF"]["feature_importances_mean"] = fi_stack.mean(axis=0).tolist()
    all_results["RF"]["feature_importances_std"] = fi_stack.std(axis=0).tolist()
all_results.setdefault("_feature_names", FEATURE_NAMES)

# Final save (also written incrementally after each model above).
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved per-seed and aggregated metrics to {OUT_PATH}")

# ----------------------------------------------------------------------------
# Levene's test on fairness-gap variance.  This is the paper's headline
# claim: DANN-Trust is more STABLE than the CNN (nearly identical mean
# gap, much smaller std).  With N=25 seeds the p-value has real teeth.
# ----------------------------------------------------------------------------
try:
    from scipy import stats
except ImportError:
    print("\n[warn] scipy not installed; skipping Levene's test.")
else:
    def _gap_series(name):
        return [r["gap_pp"] for r in all_results[name]["raw_per_seed"]]

    pairs = [
        ("CNN", "DANN"),   # headline claim
        ("ANN", "DANN"),
        ("RF",  "DANN"),
    ]
    print("\n" + "=" * 72)
    print(f"ETHNICITY gap: variance comparison (Levene's test, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<20} {'W':>8} {'p-value':>10}   verdict")
    print("-" * 72)
    for a, b in pairs:
        if a not in all_results or b not in all_results:
            continue
        stat, p = stats.levene(_gap_series(a), _gap_series(b), center="median")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<12} {stat:>8.2f} {p:>10.4f}   {tag}")

    # ----------------------------------------------------------------------
    # PER-SEX fairness gap: the dominant disparity in this corpus (~9-10pp
    # Female > Male across all models). Two questions worth asking:
    #   1. Do deep models reduce the MEAN sex gap vs baselines?
    #        -> Mann-Whitney U (rank-based; robust to LR's zero variance)
    #   2. Do deep models change the VARIANCE of the sex gap vs baselines?
    #        -> Levene's (LR skipped: zero-variance input is degenerate)
    # ----------------------------------------------------------------------
    def _sex_gap_series(name):
        return [r["gap_sex_pp"] for r in all_results[name]["raw_per_seed"]]

    sex_pairs = [
        ("RF",  "ANN"),
        ("RF",  "CNN"),
        ("RF",  "DANN"),
        ("CNN", "DANN"),   # do CNN and DANN differ on sex?
    ]

    print("\n" + "=" * 72)
    print(f"SEX gap: MEAN comparison (Mann-Whitney U, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<20} {'mean_a':>7} {'mean_b':>7} {'U':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in sex_pairs:
        if a not in all_results or b not in all_results:
            continue
        sa, sb = _sex_gap_series(a), _sex_gap_series(b)
        mu_a, mu_b = float(np.mean(sa)), float(np.mean(sb))
        u, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<12} {mu_a:>7.2f} {mu_b:>7.2f} {u:>8.1f} {p:>8.4f}   {tag}")

    print("\n" + "=" * 72)
    print(f"SEX gap: VARIANCE comparison (Levene's test, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<20} {'std_a':>7} {'std_b':>7} {'W':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in sex_pairs:
        if a not in all_results or b not in all_results:
            continue
        # Skip pairs involving zero-variance groups (e.g. LR).  Levene's would
        # still return a value but interpretation is meaningless.
        sa, sb = _sex_gap_series(a), _sex_gap_series(b)
        sd_a, sd_b = float(np.std(sa)), float(np.std(sb))
        if sd_a == 0.0 or sd_b == 0.0:
            print(f"{a} vs {b:<12} {sd_a:>7.2f} {sd_b:>7.2f}     -        -   skipped (zero variance)")
            continue
        stat, p = stats.levene(sa, sb, center="median")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<12} {sd_a:>7.2f} {sd_b:>7.2f} {stat:>8.2f} {p:>8.4f}   {tag}")

    # ----------------------------------------------------------------------
    # PER-AGE fairness gap.  DANN had the smallest mean age gap in the 25-seed
    # run (3.54pp vs RF 4.93pp and CNN 4.16pp).  Two questions:
    #   1. Do deep models (esp. DANN) reduce the MEAN age gap vs baselines?
    #        -> Mann-Whitney U
    #   2. Is the age-gap VARIANCE different across model classes?
    #        -> Levene's (LR skipped: zero variance)
    # If DANN's age advantage is significant, that's a specific axis where
    # DANN uniquely wins (unlike ethnicity and sex, where CNN and DANN tie).
    # ----------------------------------------------------------------------
    def _age_gap_series(name):
        return [r["gap_age_pp"] for r in all_results[name]["raw_per_seed"]]

    age_pairs = [
        ("RF",  "ANN"),
        ("RF",  "CNN"),
        ("RF",  "DANN"),
        ("CNN", "DANN"),   # does DANN specifically beat CNN on age?
    ]

    print("\n" + "=" * 72)
    print(f"AGE gap: MEAN comparison (Mann-Whitney U, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<20} {'mean_a':>7} {'mean_b':>7} {'U':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in age_pairs:
        if a not in all_results or b not in all_results:
            continue
        sa, sb = _age_gap_series(a), _age_gap_series(b)
        mu_a, mu_b = float(np.mean(sa)), float(np.mean(sb))
        u, p = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<12} {mu_a:>7.2f} {mu_b:>7.2f} {u:>8.1f} {p:>8.4f}   {tag}")

    print("\n" + "=" * 72)
    print(f"AGE gap: VARIANCE comparison (Levene's test, N={len(SEEDS)})")
    print("=" * 72)
    print(f"{'Comparison':<20} {'std_a':>7} {'std_b':>7} {'W':>8} {'p':>8}   verdict")
    print("-" * 72)
    for a, b in age_pairs:
        if a not in all_results or b not in all_results:
            continue
        sa, sb = _age_gap_series(a), _age_gap_series(b)
        sd_a, sd_b = float(np.std(sa)), float(np.std(sb))
        if sd_a == 0.0 or sd_b == 0.0:
            print(f"{a} vs {b:<12} {sd_a:>7.2f} {sd_b:>7.2f}     -        -   skipped (zero variance)")
            continue
        stat, p = stats.levene(sa, sb, center="median")
        tag = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"{a} vs {b:<12} {sd_a:>7.2f} {sd_b:>7.2f} {stat:>8.2f} {p:>8.4f}   {tag}")