"""
-----------------------------------------------------------------------------------------------------

The purpose of this module is to test the normality of a dataset (df).
- The larger the test statistic value is, the larger usually the deviation from a normal distribution.
- A non-significant p-value suggests that the dataset appears Gaussian / normally distributed.
- A significant p-value suggests that the dataset is considered skewed, hence a
different distribution should be considered during analyses.

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
from scipy.stats import shapiro
from Custom_packages.misc import visual_formatting_module, helpers_module


def shapiro_wilks_normality(df: pd.DataFrame, variable_names: list[str], factor_names: list[str] = []) -> None:
    """
    Conducts the Shapiro-Wilk normality test for specified variables and prints the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing the variables to test.
        variable_names (list[str]): Column names of variables to test for normality (e.g. acoustic features).
        factor_names (list[str], optional): Columns for grouping data into subsets (e.g. speaker demographics).

    Returns:
        None

    Raises:
        ValueError: If DataFrame or variable_names are None or empty.
    """

    try:
        helpers_module.validate_non_empty_and_exists(df, "df")
        helpers_module.validate_non_empty_and_exists(variable_names, "variable_names")

        for variable in variable_names:
            if factor_names:
                for factor in factor_names:
                    for value in df[factor].unique():
                        subset = df[df[factor] == value][variable]
                        test_statistic, p_value = shapiro(subset)
                        p_value = visual_formatting_module.highlight_significant_results(p_value)
                        print(f"Test statistic for {factor}={value}, {variable}: {test_statistic}, p-value: {p_value}")
            else:
                test_statistic, p_value = shapiro(df[variable])
                p_value = visual_formatting_module.highlight_significant_results(p_value)
                print(f"Test statistic for {variable}: {test_statistic}, p-value: {p_value}")

    except ValueError as e:
        visual_formatting_module.print_coloured(f"\nError during Shapiro Wilks normality testing\n: {e}", "red")

