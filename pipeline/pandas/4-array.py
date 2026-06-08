#!/usr/bin/env python3
"""Converts specific columns of a DataFrame to a NumPy array"""


def array(df):
    """
    Selects the last 10 rows of High and Close columns and converts to numpy.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
