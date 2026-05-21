"""
-----------------------------------------------------------------------------------------------------

The purpose of this module is to use a Random Forest (RF) model to validate its reliability with a
leave-one-speaker-out cross-validation approach, and to estimate Gini feature importance scores.

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from Custom_packages.misc import data_visualisation_module, helpers_module


def perform_RF_model_assessment(features: pd.DataFrame, target: pd.Series, speaker_ids: np.array,
                                    group_name="") -> None:
    """
    Performs a leave-one-speaker-out cross-validated Random Forest (RF) evaluation. Calls relevant helper functions to
    print metrics (accuracy, precision, recall, F1 score) and calls relevant functions to plot confusion matrices,
    ROC curves and Gini feature importance scores.

    Parameters:
        - features (pd.DataFrame): Independent variables.
        - target (pd.Series): Dependent variable.
        - speaker_ids (np.array): Speaker IDs for cross-validation.
        - group_name (str, optional): Name of the demographic group for labeling outputs.

    Returns:
        None
    """

    # Set arguments for reproducibility purposes.
    model = RandomForestClassifier(random_state=1, n_estimators=126)

    # Perform cross-validation to get predictions
    logo = LeaveOneGroupOut()
    y_pred_cv = cross_val_predict(model, features, target, cv=logo.split(features, target, speaker_ids))

    helpers_module.calculate_metrics(target, y_pred_cv, group_name)

    data_visualisation_module.plot_confusion_matrix(target, y_pred_cv,
                                                    f"{group_name} Speakers: Confusion Matrix - Cross-Validated Random Forest (RF) Model",
                                                    "RF")

    # Run cross-validated predictions to calculate ROC-AUC metrics that depend on prediction probability scores for each target class.
    y_pred_proba_cv = cross_val_predict(model, features, target, cv=logo.split(features, target, speaker_ids),
                                        method="predict_proba")

    data_visualisation_module.plot_roc_curve(target, y_pred_proba_cv, group_name, "RF")

    # Calculate Gini feature importances
    calculate_gini_feature_importance(features, target, model, group_name)


def calculate_gini_feature_importance(features: pd.DataFrame, target: pd.Series, model: RandomForestClassifier,
                                      group_name: str) -> None:
    """
    Estimates Gini feature importances and visualises them.

    Parameters:
        - features (pd.DataFrame): Independent variables.
        - target (pd.Series): Dependent variable.
        - model (RandomForestClassifier): Trained random forest model.
        - group_name (str): Name of the group for labeling outputs.

    Returns:
        None
    """
    feature_name_mapping = {
        "Mean_Pitch(F0)": "Mean Pitch",
        "StDev_Pitch(F0)": "SD Pitch",
        "Harmonics-to-Noise_Ratio": "HNR",
        "RAP_Jitter": "RAP Jitter",
        "apq3_shimmer": "APQ3 Shimmer",
        "cpp": "CPP",
        "Voice_Duration": "Duration",
        "LTAS_Mean(dB)": "Mean LTAS",
        "LTAS_standard_deviation_(dB)": "SD LTAS",
        "LTAS_slope_(dB)": "LTAS Slope"
    }

    model.fit(features, target)

    # Get feature importance scores, and sort them for visualisation purposes.
    gini_importance = model.feature_importances_
    feature_names = features.columns
    gini_importance_dict = dict(zip(feature_names, np.round(gini_importance * 100, 2)))
    sorted_gini_importance = sorted(gini_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Map feature names and visualise their importance scores.
    mapped_feature_names = [feature_name_mapping.get(name, name) for name, _ in sorted_gini_importance]
    importance_scores = [score for _, score in sorted_gini_importance]

    data_visualisation_module.plot_feature_importances(
        mapped_feature_names, np.array(importance_scores), "Gini", group_name
    )
