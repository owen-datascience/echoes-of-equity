"""
-----------------------------------------------------------------------------------------------------

The purpose of this module is to create a variety of plots (e.g. confusion matrices, roc curves, etc).

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


# using ANSI escape codes to highlight messages or values of interest using colours.
colours_dict = {
    "reset": "\033[0m",
    "orange": "\033[33m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "red": "\033[31m"
}

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, heading: str, model: str) -> None:
    """
    Plots a confusion matrix as a heatmap using seaborn.

    Parameters:
        y_true (pd.Series): True labels for the confusion matrix.
        y_pred (np.ndarray): Predicted labels for the confusion matrix.
        heading (str): The title for the plot.
        model (str): Model type (e.g. LR or RF), which determines the heatmap colour scheme.

    Returns:
        None
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plot_colour = "Reds"
    if model == "LR":
        plot_colour = "Blues"

    sns.heatmap(cm, annot=True, fmt='d', cmap=plot_colour, xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(heading)
    plt.show()
    #plt.close()


def plot_roc_curve(target: pd.Series, y_pred_proba_cv: np.array, group_name: str, model: str) -> None:
    """
    Plots the ROC curve for cross-validated model predictions.

    Parameters:
        - target (pd.Series): True target values.
        - y_pred_proba_cv (np.array): Predicted probability scores, used to compute ROC-AUC metrics.
        - group_name (str): Name of the group for labeling outputs.
        - model (str): Model type (e.g. LR or RF), which determines the heatmap colour scheme.

    Returns:
        None
    """

    fpr_cv, tpr_cv, _ = roc_curve(target, y_pred_proba_cv[:, 1])
    roc_auc_cv = auc(fpr_cv, tpr_cv)

    plot_colour = "red"
    if model == "LR":
        plot_colour = "blue"

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_cv, tpr_cv, color=plot_colour, lw=2, label=f'Cross-Validated ROC curve (AUC = {roc_auc_cv:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{group_name} Speakers: ROC Curve - Cross-Validated {model} Model')
    plt.legend(loc="lower right")
    plt.show()
    #plt.close()


def plot_feature_importances(feature_names: list[str], importance_scores: np.array, importance_method_name = "", speaker_group = "") -> None:
    """
    Plots the importance scores of features using horizontal bar plots.

    Parameters:
        - feature_names (list[str]): List of feature column names.
        - importance_scores (np.array): Numpy array of importance scores.
        - importance_method_name (str, optional): Name of the importance scores method used, for display purposes.
        - speaker_group (str, optional): Demographic group name, for display purposes.
    """

    # sorts importance scores and corresponding acoustic feature names in descending order (original params already sorted in an ascending order).
    sorted_indices = importance_scores.argsort()[::-1]                  # sorting the indices of the array in descending order.
    sorted_importance_scores = importance_scores[sorted_indices]        # applying the sorted indices to sort the items within our array.
    sorted_feature_names = [feature_names[i] for i in sorted_indices]   # finally, sorting our acoustic column names with the help of the sorted indices too, to match our newly sorted array items.

    # visualises them using a horizontal bar plot.
    plt.figure(figsize = (10, 6))
    plt.bar(range(len(importance_scores)), sorted_importance_scores, color = "#D10000", align = "center")
    plt.xticks(range(len(importance_scores)), sorted_feature_names, rotation = 90)

    plt.ylabel(f"{importance_method_name} importance score (%)")
    if importance_method_name == "Removed acoustic feature":
        plt.xlabel("Acoustic feature removed")
        plt.title(f"{speaker_group} speakers: {importance_method_name} importance")
    else:
        plt.xlabel("Acoustic feature")
        plt.title(f"{speaker_group} speakers: {importance_method_name} acoustic feature importance")

    # Finds the max importance score from the data, then
    # sets dynamic y-axis limit, but cap it at 20, unless the data max value is higher than 20, and in that case display the appropriate max value.
    max_importance = np.max(sorted_importance_scores)  # Find the maximum importance score
    plt.ylim(0, max(25, max_importance))

    plt.tight_layout()
    plt.show()
    #plt.close()

    return None  # (optional) we explicitly specify that there is nothing for this function to return; we could have excluded this line of code altogether.
