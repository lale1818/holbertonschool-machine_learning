#!/usr/bin/env python3
"""Prunes NaN values from a DataFrame"""


def prune(df):
    """
    Removes entries where Close has NaN values.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The modified dataframe.
    """
    return df.dropna(subset=['Close'])
