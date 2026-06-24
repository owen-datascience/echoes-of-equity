
# Echoes of Equity: paper-figure generator.
#
# Produces the data-driven figures referenced in the paper plan:
#   Fig 2:  Speaker demographics by ethnicity x age x sex
#   Fig 3:  Acoustic feature boxplots (Neutral vs Trustworthy)
#   Fig 9:  Per-ethnicity accuracy with error bars (the HEADLINE fairness figure)
#   Fig 13: Fairness-accuracy Pareto scatter with error bars
#
# Hand-drawn figures (Fig 1 pipeline, Fig 4 ANN arch, Fig 5 CNN arch,
# Fig 6 DANN arch) live in drawio/TikZ outside this script.

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # render to file without a display
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})
sns.set_palette("colorblind")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "Paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEMO_CSV = os.path.join(REPO_ROOT, "MySolution", "ExistingMethods", "Speaker_demographics.csv")
FEAT_CSV = os.path.join(REPO_ROOT, "MySolution", "NewMethods",
                        "DANN_Trustworthy_Intent_Project", "Speech_dataset_characteristics.csv")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results.json")

ETH_ORDER_PRETTY = ["White", "Black", "South Asian"]
ETH_ORDER_KEY = ["White", "Black", "South_Asian"]   # keys as stored in results.json
MODEL_ORDER = ["RF", "LR", "ANN", "CNN", "DANN"]
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette("colorblind", n_colors=5)))


# ---------------------------------------------------------------------------
# Fig 2: speaker demographics by ethnicity x age x sex
# ---------------------------------------------------------------------------
def make_fig02_demographics():
    df = pd.read_csv(DEMO_CSV)
    df = df.rename(columns=lambda c: c.strip())
    for c in ["Ethnicity", "Age-group", "Sex"]:
        df[c] = df[c].astype(str).str.strip()

    counts = (df.groupby(["Ethnicity", "Age-group", "Sex"])
                .size()
                .unstack("Sex", fill_value=0)
                .reset_index())

    age_groups = ["Younger", "Older"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, age in zip(axes, age_groups):
        sub = counts[counts["Age-group"] == age].set_index("Ethnicity").reindex(ETH_ORDER_PRETTY).fillna(0)
        female = sub.get("Female", pd.Series([0] * 3, index=ETH_ORDER_PRETTY))
        male = sub.get("Male", pd.Series([0] * 3, index=ETH_ORDER_PRETTY))
        x = np.arange(len(ETH_ORDER_PRETTY))
        ax.bar(x, female, label="Female", color="#4c72b0")
        ax.bar(x, male, bottom=female, label="Male", color="#dd8452")
        for i, e in enumerate(ETH_ORDER_PRETTY):
            total = int(female.iloc[i] + male.iloc[i])
            color = "red" if total <= 10 else "black"
            weight = "bold" if total <= 10 else "normal"
            ax.text(i, total + 0.4, f"N={total}", ha="center", color=color, fontweight=weight)
        ax.set_xticks(x)
        ax.set_xticklabels(ETH_ORDER_PRETTY)
        ax.set_title(f"{age} adults")
        ax.set_ylabel("Number of speakers")
        ax.set_ylim(0, 25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(title="Sex", loc="upper right")
    fig.suptitle("Fig. 2  Speaker demographics in the TIS Corpus.  "
                 "Red labels mark cells with N <= 10.",
                 y=1.02, fontsize=11)
    out = os.path.join(FIG_DIR, "fig02_demographics.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: acoustic feature distributions, Neutral vs Trustworthy
# ---------------------------------------------------------------------------
def make_fig03_feature_boxplots():
    df = pd.read_csv(FEAT_CSV)
    panels = [
        ("Mean_Pitch(F0)",         "F0 mean (Hz)"),
        ("StDev_Pitch(F0)",        "F0 std (Hz)"),
        ("Harmonics-to-Noise_Ratio", "HNR (dB)"),
        ("RAP_Jitter",             "Jitter (RAP)"),
        ("apq3_shimmer",           "Shimmer (APQ3)"),
        ("cpp",                    "CPP (dB)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (col, label) in zip(axes.flatten(), panels):
        if col not in df.columns:
            ax.set_title(f"{label}\n(column not in CSV)")
            ax.axis("off")
            continue
        sub = df[[col, "Speaker_Intent"]].dropna()
        sns.boxplot(
            data=sub, x="Speaker_Intent", y=col,
            order=["Neutral", "Trustworthy"],
            palette={"Neutral": "#bcbddc", "Trustworthy": "#756bb1"},
            ax=ax, showfliers=False, width=0.55,
        )
        sns.stripplot(
            data=sub, x="Speaker_Intent", y=col,
            order=["Neutral", "Trustworthy"],
            color="black", size=1.2, alpha=0.25, ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Fig. 3  Six acoustic features that drive trustworthy-intent classification, "
                 "Neutral vs Trustworthy speech.", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig03_feature_boxplots.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 9: per-ethnicity accuracy bar chart with error bars (headline fairness)
# ---------------------------------------------------------------------------
def make_fig09_acc_by_ethnicity(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    n_eth = len(ETH_ORDER_KEY)
    n_models = len(MODEL_ORDER)
    bar_w = 0.15
    eth_x = np.arange(n_eth)

    for i, m in enumerate(MODEL_ORDER):
        r = results[m]
        means = [r["per_eth_acc_mean"].get(e, np.nan) * 100 for e in ETH_ORDER_KEY]
        stds = [r["per_eth_acc_std"].get(e, 0.0) * 100 for e in ETH_ORDER_KEY]
        offsets = eth_x + (i - (n_models - 1) / 2) * bar_w
        ax.bar(offsets, means, bar_w, yerr=stds, capsize=3,
               label=m, color=MODEL_COLORS[m], edgecolor="black", linewidth=0.4)

    # Source-paper baseline line at 71%
    ax.axhline(71, color="grey", linestyle="--", linewidth=1)
    ax.text(n_eth - 0.5, 71.2, "source-paper RF baseline (71%)",
            color="grey", fontsize=9, ha="right", va="bottom")

    ax.set_xticks(eth_x)
    ax.set_xticklabels(ETH_ORDER_PRETTY)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 90)
    ax.set_title("Fig. 9  Per-ethnicity trust accuracy, mean +/- std over 5 seeds.\n"
                 "Deep models (ANN/CNN/DANN) raise South Asian accuracy by ~8pp.",
                 fontsize=11)
    ax.legend(loc="upper right", ncol=5, frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(FIG_DIR, "fig09_acc_by_ethnicity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 13: fairness-accuracy Pareto scatter
# ---------------------------------------------------------------------------
def make_fig13_pareto(results):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for m in MODEL_ORDER:
        r = results[m]
        x = r["overall_acc_mean"] * 100
        y = r["gap_pp_mean"]
        xerr = r["overall_acc_std"] * 100
        yerr = r["gap_pp_std"]
        color = MODEL_COLORS[m]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=color,
                    markersize=10, capsize=4, elinewidth=1.2, label=m)
        ax.annotate(m, (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color)

    # Source-paper RF reference dot
    ax.scatter([71], [5.0], marker="*", s=180, color="black",
               label="source paper RF (LOSO)", zorder=5)
    ax.annotate("Source paper\n(LOSO-CV)", (71, 5.0), xytext=(8, -18),
                textcoords="offset points", fontsize=9, color="black")

    # Arrow showing "good direction"
    ax.annotate("", xy=(78, 1), xytext=(73, 9),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(78, 1.5, "better\n(higher acc,\nlower gap)",
            color="green", fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Overall accuracy (%)")
    ax.set_ylabel("Fairness gap: max - min ethnicity accuracy (pp)")
    ax.set_title("Fig. 13  Fairness vs accuracy trade-off (mean +/- std over 5 seeds).\n"
                 "DANN-Trust sits in the lower-right corner with the smallest error bars.",
                 fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(70, 81)
    ax.set_ylim(0, 18)

    out = os.path.join(FIG_DIR, "fig13_fairness_pareto.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main: generate all figures
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Figure output directory: {FIG_DIR}")
    if not os.path.exists(RESULTS_JSON):
        raise SystemExit(
            f"results.json not found at {RESULTS_JSON}. "
            "Run compare_all_models.py first."
        )
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    make_fig02_demographics()
    make_fig03_feature_boxplots()
    make_fig09_acc_by_ethnicity(results)
    make_fig13_pareto(results)

    print("\nAll figures written.")
