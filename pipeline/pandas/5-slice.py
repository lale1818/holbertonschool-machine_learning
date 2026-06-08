#!/usr/bin/env python3
"""Slices specific columns and rows from a DataFrame"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row.
    """
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    return df[columns].iloc[::60]
