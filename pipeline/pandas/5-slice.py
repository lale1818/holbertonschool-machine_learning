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
    # Lazımi sütunları seçirik və .iloc[::60] ilə hər 60-cı sətri götürürük
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
