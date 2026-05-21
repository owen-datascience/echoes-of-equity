
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def main():
    base = Path(__file__).resolve().parents[1]
    data = np.load(base/"data/synthetic_preds.npz")
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve (AUC={roc_auc:.2f})")
    out = base/"data/roc_curve.png"
    plt.savefig(out)
    print("Saved ROC to:", out)

if __name__ == "__main__":
    main()
