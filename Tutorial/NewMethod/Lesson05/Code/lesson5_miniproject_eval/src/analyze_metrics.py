"""
Lesson 5 Mini Project: Evaluation & Fairness Metrics

This script:
1. Loads a synthetic dataset of true labels, scores, predictions, and ethnicity.
2. Computes confusion matrix, precision, recall, F1, and ROC-AUC overall.
3. Computes per-group metrics and a fairness gap in recall.
4. Plots an ROC curve.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
)

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "synthetic_eval_results.csv"
    df = pd.read_csv(data_path)
    return df

def overall_metrics(df):
    print("=== Overall Confusion Matrix (threshold = 0.5) ===")
    cm = confusion_matrix(df["y_true"], df["y_pred"])
    print(cm)
    print("\n=== Overall Classification Report ===")
    print(classification_report(df["y_true"], df["y_pred"], target_names=["neutral","trustworthy"]))

    fpr, tpr, _ = roc_curve(df["y_true"], df["score_trustworthy"])
    roc_auc = auc(fpr, tpr)
    print(f"Overall ROC-AUC: {roc_auc:.3f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC Curve")
    plt.legend(loc="lower right")
    base_dir = Path(__file__).resolve().parents[1]
    out_path = base_dir / "data" / "roc_overall.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved ROC curve to: {out_path}")

def group_metrics(df):
    print("\n=== Metrics by Ethnicity ===")
    recalls = {}
    for group, sub in df.groupby("ethnicity"):
        y_true = sub["y_true"]
        y_pred = sub["y_pred"]
        rec = recall_score(y_true, y_pred, pos_label=1)
        prec = precision_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        recalls[group] = rec
        print(f"\nGroup: {group}")
        print(f"  Precision (trustworthy): {prec:.3f}")
        print(f"  Recall (trustworthy):    {rec:.3f}")
        print(f"  F1 (trustworthy):        {f1:.3f}")

    max_rec = max(recalls.values())
    min_rec = min(recalls.values())
    gap = max_rec - min_rec
    print(f"\nFairness gap in recall (max - min): {gap:.3f}")

def main():
    df = load_data()
    print(df.head())
    overall_metrics(df)
    group_metrics(df)

if __name__ == "__main__":
    main()
