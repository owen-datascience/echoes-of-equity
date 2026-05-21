"""
-----------------------------------------------------------------------------------------------------

This module includes common utilities used across different ML and stats scripts, such as data
preparation and validation, task routing (i.e. directing to appropriate functions) and model evaluation metrics.
These utilities are designed to support larger workflows and are not meant for standalone use.

-----------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Custom_packages.misc import visual_formatting_module
# the machine_learning package is needed for the analysis_func parameter.
from Custom_packages.machine_learning import random_forest_module, regression_ML_module
from Custom_packages.stats import regression_stats_module


def validate_non_empty_and_exists(param, param_name="parameter") -> None:
    """
    Ensures the parameter exists and is not empty (for strings, lists, or DataFrames).

    Parameters:
        param: The input to be validated; can be any type.
        param_name (str): The name of the parameter, used in error messages.

    Returns:
        None

    Raises:
        ValueError: If the parameter does not exist or is empty.
    """

    if isinstance(param, (pd.DataFrame, pd.Series, np.ndarray)):
        if param.empty:
            raise ValueError(f"The '{param_name}' parameter either does not exist or is empty.")
    elif not param:
        raise ValueError(f"The '{param_name}' parameter either does not exist or is empty.")


def stratified_model_evaluation_helper(strata_dict: dict, variables_dict: dict, analysis_func: callable, analysis_label: str) -> None:
    """
    Prepares the data and calls the relevant functions for model cross-validation analyses. It also prints the results
    for each speaker demographic strata. Specifically, it focuses on ethnicity and age-group strata.

    Parameters:
        strata_dict (dict): A dictionary where the key is the strata label (e.g., speaker ethnicity or age group)
                             and the value is the corresponding DataFrame.
        variables_dict (dict): A dictionary where the first key is named "predictors" and holds the predictor variables
                               (e.g., acoustic features), and the second key is named "target" and holds a single
                               target variable.
        analysis_func (callable): A function that performs the analysis (e.g., model evaluation).
        analysis_label (str): A label to identify the type of analysis (e.g., "Logistic Regression Analysis").

    Returns:
        None
    """

    try:
        validate_non_empty_and_exists(strata_dict, "strata_dict")
        validate_non_empty_and_exists(variables_dict, "variables_dict")
        validate_non_empty_and_exists(analysis_func, "analysis_func")
        validate_non_empty_and_exists(analysis_label, "analysis_label")

        for key, df in list(strata_dict.items()):
            visual_formatting_module.print_section_heading(f"{key} speakers: {analysis_label}")

            X = df[variables_dict["predictors"]]
            y = df[variables_dict["target"]]
            speaker_ids_array = df["Speaker_ID"]

            # Call the relevant cross-validation analysis function.
            analysis_func(X, y, speaker_ids_array, key)

            # If the analysis is LR, calculate and print the logistic regression summary to get the significance /
            # importance per acoustic feature.
            if analysis_label == "Logistic Regression Analysis":
                regression_stats_module.calculate_logistic_regression(X, y, key)

    except ValueError as e:
        visual_formatting_module.print_coloured(f"\nError during categorical encoding\n: {e}", "red")



def calculate_metrics(target: pd.Series, y_pred_cv: np.array, group_name: str) -> None:
    """
    Calculates and prints accuracy, precision, recall, and F1 score for cross-validated (CV) model.

    Parameters:
        - target (pd.Series): True target values.
        - y_pred_cv (np.array): Predicted target values from cross-validation.
        - group_name (str): Name of the group for labeling outputs.

    Returns:
        None
    """
    metrics_cv = {
        "Accuracy": accuracy_score(target, y_pred_cv),
        "Precision": precision_score(target, y_pred_cv),
        "Recall": recall_score(target, y_pred_cv),
        "F1 Score": f1_score(target, y_pred_cv)
    }

    print(f"\n{group_name} Speakers - Cross-Validated Model Metrics:")
    for metric, value in metrics_cv.items():
        print(f"{metric}: {value:.4f}")
