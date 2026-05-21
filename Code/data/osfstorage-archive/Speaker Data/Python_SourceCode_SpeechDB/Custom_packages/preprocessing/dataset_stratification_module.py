"""
-----------------------------------------------------------------------------------------------------

The purpose of this module is to stratify an existing dataset based on specific criteria.

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
from Custom_packages.misc import helpers_module, visual_formatting_module


def create_strata(df: pd.DataFrame, criteria: list[str]) -> dict:
    """
    Splits the dataset into smaller strata based on the provided criteria columns (e.g., ethnicity, age-group).

    Parameters:
        df (pd.DataFrame): The DataFrame to be stratified, containing the columns that are to be used as criteria.
        criteria (list[str]): A list of column names used to stratify the DataFrame.

    Returns:
        dict: A dictionary containing the stratified DataFrames.
            - "All" (pd.DataFrame): The original DataFrame, included in the result as a key for the entire dataset.
            - {group_value} (pd.DataFrame): DataFrames grouped by the unique values in the specified criteria columns.

    Raises:
        ValueError: If the DataFrame or criteria parameters are None or empty.
    """
    try:
        helpers_module.validate_non_empty_and_exists(df, "df")
        helpers_module.validate_non_empty_and_exists(criteria, "criteria")

        strata_dict = {'All': df}

        # Iterate through the criteria columns and create strata based on each
        for column_name in criteria:
            grouped = df.groupby(column_name)

            for key, group in grouped:
                strata_dict[key] = group.copy()

        return strata_dict
    except ValueError as e:
        visual_formatting_module.print_coloured(f"\nError during categorical encoding\n: {e}", "red")



