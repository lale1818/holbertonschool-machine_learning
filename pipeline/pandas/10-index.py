#!/usr/bin/env python3
"""Defines a function to set a specific column as the index of a DataFrame"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The modified dataframe with Timestamp as index.
    """
    return df.set_index('Timestamp')
