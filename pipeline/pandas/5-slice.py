#!/usr/bin/env python3
"""
Defines a function that slices specific columns from a pd.DataFrame
"""
import pandas as pd


def slice(df):
    """
    Extracts High, Low, Close, and Volume_(BTC) columns
    and selects every 60th row
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
