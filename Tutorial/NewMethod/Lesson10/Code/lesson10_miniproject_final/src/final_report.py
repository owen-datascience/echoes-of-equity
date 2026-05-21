"""
Lesson 10 Mini Project: Final Evaluation & Report Generator

This script:
1. Loads a small synthetic evaluation dataset with:
   - demographic groups
   - ground-truth labels
   - baseline model predictions
   - debiased model predictions
2. Computes:
   - overall accuracy
   - per-group recall
   - recall fairness gap (max difference between groups)
3. Prints a concise text report comparing baseline vs debiased models.
4. Writes a markdown file "final_report.md" summarizing results.

Run with:
    python src/final_report.py
"""

from pathlib import Path
import pandas as pd

def compute_metrics(df, pred_col):
    # Overall accuracy
    acc = (df["y_true"] == df[pred_col]).mean()

    # Per-group recall for the positive class (1 = trustworthy)
    recalls = {}
    for g in sorted(df["group"].unique()):
        subset = df[df["group"] == g]
        # True positives / all actual positives in that group
        positives = subset[subset["y_true"] == 1]
        if len(positives) == 0:
            recalls[g] = float("nan")
        else:
            tp = (positives[pred_col] == 1).mean()
            recalls[g] = tp

    # Fairness gap = max difference in recall across groups (ignore NaNs)
    valid_recalls = [v for v in recalls.values() if v == v]
    if valid_recalls:
        fairness_gap = max(valid_recalls) - min(valid_recalls)
    else:
        fairness_gap = float("nan")

    return acc, recalls, fairness_gap

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "final_eval_data.csv"
    df = pd.read_csv(data_path)

    acc_base, rec_base, gap_base = compute_metrics(df, "y_pred_baseline")
    acc_deb, rec_deb, gap_deb = compute_metrics(df, "y_pred_debiased")

    print("=== Final Evaluation: Baseline vs Debiased Model ===")
    print("\nBaseline model:")
    print(f"  Overall accuracy: {acc_base:.3f}")
    print("  Per-group recall (trustworthy):")
    for g, r in rec_base.items():
        print(f"    {g:10s}: {r:.3f}")
    print(f"  Recall fairness gap: {gap_base:.3f}")

    print("\nDebiased model:")
    print(f"  Overall accuracy: {acc_deb:.3f}")
    print("  Per-group recall (trustworthy):")
    for g, r in rec_deb.items():
        print(f"    {g:10s}: {r:.3f}")
    print(f"  Recall fairness gap: {gap_deb:.3f}")

    # Write markdown report
    report_path = base_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Report (Synthetic Demo)\n\n")
        f.write("This report compares a baseline model vs a debiased model on a\n")
        f.write("small synthetic dataset with three demographic groups.\n\n")

        f.write("## Baseline Model\n\n")
        f.write(f"- Overall accuracy: **{acc_base:.3f}**\n")
        f.write(f"- Recall fairness gap (max group difference): **{gap_base:.3f}**\n\n")
        f.write("| Group | Recall (trustworthy) |\n")
        f.write("|-------|-----------------------|\n")
        for g, r in rec_base.items():
            f.write(f"| {g} | {r:.3f} |\n")
        f.write("\n")

        f.write("## Debiased Model\n\n")
        f.write(f"- Overall accuracy: **{acc_deb:.3f}**\n")
        f.write(f"- Recall fairness gap (max group difference): **{gap_deb:.3f}**\n\n")
        f.write("| Group | Recall (trustworthy) |\n")
        f.write("|-------|-----------------------|\n")
        for g, r in rec_deb.items():
            f.write(f"| {g} | {r:.3f} |\n")
        f.write("\n")

        f.write("## Interpretation\n\n")
        f.write(
            "You can edit this section to describe how the debiased model\n"
            "compares to the baseline. For example, if the debiased model has\n"
            "a similar accuracy but a smaller fairness gap, you can write that\n"
            "it maintains strong performance while improving fairness between\n"
            "demographic groups.\n"
        )

    print(f"\nMarkdown report written to: {report_path}")

if __name__ == '__main__':
    main()
