"""
-----------------------------------------------------------------------------------------------------

The purpose of this module is to conduct classification and/or predictions using regression models
with the machine learning package (sklearn from scikit-learn).

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_predict
import matplotlib.pyplot as plt
from Custom_packages.misc import data_visualisation_module, helpers_module


def perform_LR_model_assessment(features: pd.DataFrame, target: pd.Series, speaker_ids: np.array,
                                    group_name="") -> None:
    """
    Performs a leave-one-speaker-out cross-validated (LOSO CV) Logistic Regression (LR) evaluation (scikit-learn package).
    Calls relevant helper functions to print metrics (accuracy, precision,recall, F1 score) and plot confusion matrices
    and ROC curves.

    Parameters:
        - features (pd.DataFrame): Independent variables.
        - target (pd.Series): Dependent variable.
        - speaker_ids (np.array): Speaker IDs for cross-validation.
        - group_name (str, optional): Name of the demographic group for labeling outputs.

    Returns:
        None
    """

    # Set the random_state argument for reproducibility purposes.
    model = LogisticRegression(random_state=1)

    # performs cross-validated predictions and calculate metrics.
    logo = LeaveOneGroupOut()
    y_pred_cv = cross_val_predict(model, features, target, cv=logo.split(features, target, speaker_ids))

    # Calculate evaluation metrics.
    helpers_module.calculate_metrics(target, y_pred_cv, group_name)

    data_visualisation_module.plot_confusion_matrix(target, y_pred_cv,
                              f"{group_name} Speakers: Confusion Matrix - Cross-Validated Logistic Regression (LR) Model", "LR")

    # Run cross-validated predictions to calculate ROC-AUC metrics that depend on prediction probability scores for each target class.
    y_pred_proba_cv = cross_val_predict(model, features, target, cv=logo.split(features, target, speaker_ids),
                                        method="predict_proba")

    data_visualisation_module.plot_roc_curve(target, y_pred_proba_cv, group_name, "LR")
