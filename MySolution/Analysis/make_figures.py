import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
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


def make_fig2_feature_boxplots():
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

    fig.suptitle("Six acoustic features that drive trustworthy-intent classification, "
                 "Neutral vs Trustworthy speech.", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_feature_boxplots.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig6_acc_by_ethnicity(results):
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

    ax.axhline(71, color="grey", linestyle="--", linewidth=1)
    ax.text(n_eth - 0.5, 71.2, "source-paper RF baseline (71%)",
            color="grey", fontsize=9, ha="right", va="bottom")

    ax.set_xticks(eth_x)
    ax.set_xticklabels(ETH_ORDER_PRETTY)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 90)
    ax.set_title("Per-ethnicity trust accuracy, mean +/- std over 50 seeds.\n"
                 "Deep models (ANN/CNN/DANN) raise South Asian accuracy by ~8-10 pp "
                 "over Random Forest, largely closing the baseline gap.",
                 fontsize=11)
    ax.legend(loc="upper right", ncol=5, frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(FIG_DIR, "fig6_acc_by_ethnicity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig7_pareto(results):
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

    ax.scatter([71], [5.0], marker="*", s=180, color="black",
               label="source paper RF (LOSO)", zorder=5)
    ax.annotate("Source paper\n(LOSO-CV)", (71, 5.0), xytext=(8, -18),
                textcoords="offset points", fontsize=9, color="black")

    ax.annotate("", xy=(78, 1), xytext=(73, 9),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(78, 1.5, "better\n(higher acc,\nlower gap)",
            color="green", fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Overall accuracy (%)")
    ax.set_ylabel("Fairness gap: max - min ethnicity accuracy (pp)")
    ax.set_title("Fairness vs accuracy trade-off (mean +/- std over 50 seeds).\n"
                 "CNN and DANN-Trust occupy essentially the same region of the plane; "
                 "their error bars overlap on both axes.",
                 fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(70, 81)
    ax.set_ylim(0, 18)

    out = os.path.join(FIG_DIR, "fig7_fairness_pareto.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")




if __name__ == "__main__":
    print(f"Figure output directory: {FIG_DIR}")
    if not os.path.exists(RESULTS_JSON):
        raise SystemExit(
            f"results.json not found at {RESULTS_JSON}. "
            "Run compare_all_models.py first."
        )
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    make_fig2_feature_boxplots()
    make_fig6_acc_by_ethnicity(results)
    make_fig7_pareto(results)

    print("\nAll figures written.")
