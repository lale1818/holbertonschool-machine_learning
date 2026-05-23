#!/usr/bin/env python3
"""
Defines high function
"""


def high(df):
    """
    Sorts dataframe by the High price in descending order
    """
    return df.sort_values(by='High', ascending=False)
