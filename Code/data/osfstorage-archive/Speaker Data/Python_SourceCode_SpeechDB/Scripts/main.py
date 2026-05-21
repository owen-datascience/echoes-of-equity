"""
-----------------------------------------------------------------------------------------------------

This is the main script to perform the speech dataset analyses described below.

1. Pre-process the data for analysis:
    1.1. Encode categorical variables into numerical format.
    1.2. Stratify the dataset by speaker demographics (e.g., ethnicity, age group).

2. Calculate descriptive statistics for speaker demographics and acoustic features.

3. Evaluate the distribution of the data.

4. Assess the importance of acoustic features, and reliability of the dataset using:
    4.1. Classification methods (i.e., cross-validated Random Forest and Logistic Regression from the sklearn package).
    4.2. Feature importance (sklearn and statsmodels packages).

5. Codebase development and Licensing:
    5.1. This codebase supports the analyses for the upcoming publication "Human voices communicating trustworthy intent: A
    demographically diverse speech audio dataset" by Constantina Maltezou-Papastylianou, Reinhold Scherer and Silke
    Paulmann.
    5.2. Developed by: Constantina Maltezou-Papastylianou.
    5.3. Licensed under the wider project license: CC BY 4.0 International License.

-----------------------------------------------------------------------------------------------------
"""

"""
=====================================================================================================

Loading necessary package dependencies and script settings.

=====================================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

from Custom_packages.misc import data_visualisation_module, visual_formatting_module, helpers_module
from Custom_packages.preprocessing import feature_encoding_module, dataset_stratification_module
from Custom_packages.stats import descriptives_module, normality_tests_module, regression_stats_module
from Custom_packages.machine_learning import random_forest_module, regression_ML_module

pd.set_option("display.max_columns", None)  # prevent truncating columns; display all available columns when printing.
pd.set_option("display.width", None)  # widen the narrow (by default) terminal display of PyCharm IDE (set a "None" value for width auto-detection, or set the max width manually (e.g. 400); it adjusts the width based on the number of characters.


"""
=====================================================================================================

1. DATA PRE-PROCESSING

This section handles the pre-processing of the main data for the analyses. It includes encoding categorical 
variables into numerical values and stratifying the dataset based on speaker demographics and intent.

=====================================================================================================
"""

visual_formatting_module.print_section_heading("Data Pre-processing")

# Define the path to the CSV file containing speech dataset characteristics.
csv_file_path = os.path.join(os.path.abspath('..'), 'Data', 'Speech_dataset_characteristics.csv')
original_speech_dataset_df = pd.read_csv(csv_file_path)

categories_of_interest_dict = {
    "acoustics_column_names": [
        "Voice_Duration", "Mean_Pitch(F0)", "StDev_Pitch(F0)",
        "Harmonics-to-Noise_Ratio", "RAP_Jitter", "apq3_shimmer", "cpp",
        "LTAS_Mean(dB)", "LTAS_standard_deviation_(dB)", "LTAS_slope_(dB)"
    ],
    "speaker_column_names": [
        "Speaker_Ethnicity", "Speaker_AgeGroup", "Speaker_Sex", "Speaker_Intent",
        "Speaker_Ethnicity_Encoded", "Speaker_AgeGroup_Encoded", "Speaker_Sex_Encoded",
        "Speaker_Intent_Encoded"
    ]
}

# Perform label encoding (convert categorical variables to numerical values).
try:
    original_speech_dataset_df = feature_encoding_module.categorical_to_numerical(
        original_speech_dataset_df,
        categories_of_interest_dict["speaker_column_names"][:4])
except ValueError as e:
    visual_formatting_module.print_coloured(f"\nError during categorical encoding\n: {e}", "red")


# Stratify the dataset based on speakers' ethnicity and age-group.
try:
    ethnicity_age_stratified_subsets = dataset_stratification_module.create_strata(original_speech_dataset_df, categories_of_interest_dict["speaker_column_names"][:2])
except ValueError as e:
    visual_formatting_module.print_coloured(f"\nError during categorical encoding\n: {e}", "red")


"""
=====================================================================================================

2. DESCRIPTIVE STATISTICS OF SPEAKER DEMOGRAPHICS AND ACOUSTIC FEATURES

This section calculates and displays the descriptive statistics for speaker demographics and 
acoustic features.

=====================================================================================================
"""

csv_file_path = os.path.join(os.path.abspath('..'), 'Data', 'Speaker_demographics.csv')
speaker_demog_df = pd.read_csv(csv_file_path)

visual_formatting_module.print_section_heading("Speaker Demographics")

descriptive_stats = descriptives_module.get_demographic_descriptives(speaker_demog_df)

for demog_category, stats in descriptive_stats.items():
    print(f"\n{demog_category}:\n")
    print(stats)


# Get descriptive stats for each acoustic feature by speaker group.
descriptive_stats = descriptives_module.get_acoustics_descriptives(original_speech_dataset_df, categories_of_interest_dict["speaker_column_names"][:4], categories_of_interest_dict["acoustics_column_names"])

print(f"\nDescriptive statistics for acoustic features by group:")
print(descriptive_stats)


"""
=====================================================================================================

3. STATISTICAL TESTING FOR NORMALITY

This section performs statistical tests to assess the normality of the speech audio dataset.

=====================================================================================================
"""

visual_formatting_module.print_section_heading("Shapiro-Wilk Test of Normality Results")
selected_speaker_columns = categories_of_interest_dict["speaker_column_names"][:2] + [categories_of_interest_dict["speaker_column_names"][3]]

normality_tests_module.shapiro_wilks_normality(
    original_speech_dataset_df,
    categories_of_interest_dict["acoustics_column_names"],
    selected_speaker_columns
)


"""
=====================================================================================================

4. BEGINNING OF ANALYSES

This section performs classification analyses using Random Forest and Logistic Regression. 
The analysis also examines the importance of acoustic features (using Gini importance) 
and cross-validates the robustness of the models.

=====================================================================================================
"""

variables_dict = {
    "predictors": categories_of_interest_dict["acoustics_column_names"],
    "target": categories_of_interest_dict["speaker_column_names"][7] # Encoded speaker intent column.
}

# Random Forest Analysis (including Gini Feature Importance).
helpers_module.stratified_model_evaluation_helper(ethnicity_age_stratified_subsets, variables_dict, random_forest_module.perform_RF_model_assessment, "Random Forest Analysis")

print("\n\nFinished RF analyses!\n")

# Logistic Regression (LR) Analysis.
helpers_module.stratified_model_evaluation_helper(ethnicity_age_stratified_subsets, variables_dict, regression_ML_module.perform_LR_model_assessment, "Logistic Regression Analysis")

print("\n\nFinished all analyses!")