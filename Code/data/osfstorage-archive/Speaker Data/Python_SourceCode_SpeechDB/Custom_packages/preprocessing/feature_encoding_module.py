"""
-----------------------------------------------------------------------------------------------------

This module encodes categorical variables of interest into numerical values and stores them as
new columns in the provided dataframe.

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Custom_packages.misc import helpers_module, visual_formatting_module


def categorical_to_numerical(df: pd.DataFrame, categorical_variable_columns: list[str]) -> pd.DataFrame:
    """
    Converts categorical variables into numerical values and stores them in new columns within the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical data.
        categorical_variable_columns (List[str]): A list of column names to encode as numerical values.

    Returns:
        pd.DataFrame: The updated DataFrame with encoded columns.

    Raises:
        ValueError: If the DataFrame or the list of categorical columns are None or empty.

    Example:
        df_encoded = categorical_to_numerical(df, ["Speaker_Ethnicity", "Speaker_AgeGroup"])
    """
    try:
        helpers_module.validate_non_empty_and_exists(df, "df")
        helpers_module.validate_non_empty_and_exists(categorical_variable_columns, "categorical_variable_columns")

        label_encoder = LabelEncoder()
        for column in categorical_variable_columns:
            # Create a new column for the encoded values with '_Encoded' suffix
            df[column + "_Encoded"] = label_encoder.fit_transform(df[column])

        return df
    except ValueError as e:
        visual_formatting_module.print_coloured(f"\nError during categorical encoding\n: {e}", "red")
