"""
-----------------------------------------------------------------------------------------------------

This module provides visual formatting utilities of script outputs, such as:
    - Print clearly formatted section headings for organised outputs.
    - Apply color coding to messages using ANSI escape codes.
    - Highlight statistical results (e.g., p-values) with significance indicators.

These functions are designed to improve the readability and presentation of data.

-----------------------------------------------------------------------------------------------------
"""

# Loading necessary package dependencies.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


# using ANSI escape codes to highlight messages or values of interest using colours.
colours_dict = {
    "reset": "\033[0m",
    "orange": "\033[33m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "red": "\033[31m"
}

def print_section_heading(heading: str) -> None:
    """
    Prints a formatted section heading to visually separate script sections.

    Parameters:
        heading (str): The text to display as the section heading.

    Returns:
        None

    Example:
        >>> print_section_heading("Data Pre-processing")
        ===========================================================
        Data Pre-processing
        ===========================================================
    """
    print("\n\n" + "=" * 75)
    print(heading)
    print("=" * 75 + "\n\n")

def print_coloured(message: str, colour: str) -> None:
    """
    Prints a message with a specified colour.

    Parameters:
        message (str): The message to be printed.
        color (str): The colour in which to print the message.
                     Must be one of the keys in `colours_dict`.

    Returns:
        None
    """
    print(f"{colours_dict[colour]}{message}{colours_dict['reset']}")


def highlight_significant_results(p_value: float) -> str:
    """
    Format a p-value with significance indicators.

    Parameters:
        p_value (float): The p-value to evaluate.

    Returns:
        str: The p-value rounded to three decimals:
            - Green with '**' if p-value ≤ 0.001.
            - Green with '*' if p-value ≤ 0.05.
            - Plain text otherwise.
    """
    if p_value <= 0.001:
        return f"{colours_dict['green']}{p_value:.3f} **{colours_dict['reset']}"
    elif p_value <= 0.05:
        return f"{colours_dict['green']}{p_value:.3f} *{colours_dict['reset']}"
    else:
        return f"{p_value:.3f}"