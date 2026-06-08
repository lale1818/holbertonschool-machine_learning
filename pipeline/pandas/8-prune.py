#!/usr/bin/env python3
"""Defines a function to prune NaN values from a specific column in a DataFrame"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The modified dataframe without NaN values in Close column.
    """
    return df.dropna(subset=['Close'])
