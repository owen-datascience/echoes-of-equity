"""
-----------------------------------------------------------------------------------------------------

This module is meant to calculate and return the descriptive statistics for speaker demographics
and acoustic features.

-----------------------------------------------------------------------------------------------------
"""

import os
import pandas as pd
from Custom_packages.misc import visual_formatting_module


def get_demographic_descriptives(df: pd.DataFrame) -> dict:
    """
    Calculates various descriptive statistics for speaker demographics.

    Parameters:
        df (pd.DataFrame): The dataframe containing the speaker demographics data.

    Returns:
        dict: A dictionary containing value counts and groupings for different demographic categories.
    """
    descriptive_stats = {
        "Ethnicity demographics": df["Ethnicity"].value_counts(),
        "Age-Group demographics": df.groupby("Ethnicity")["Age-group"].value_counts(),
        "Sex demographics": df.groupby(["Ethnicity", "Age-group"])["Sex"].value_counts(),
        "Age demographics": df.groupby(["Ethnicity", "Age-group", "Sex"])["Age"].describe()
    }
    return descriptive_stats


def get_acoustics_descriptives(df: pd.DataFrame, strata_columns: list, acoustic_columns: list) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each acoustic feature per stratified speaker group.

    Parameters:
        df (pd.DataFrame): The dataframe containing the dataset with acoustic features.
        strata_columns (list): List of speaker columns to group by for stratification.
        acoustic_columns (list): List of acoustic feature columns to calculate statistics for.

    Returns:
        pd.DataFrame: A DataFrame with calculated descriptive statistics (mean, std) for each speaker group.
    """
    return df.groupby(strata_columns)[acoustic_columns].agg(['mean', 'std'])