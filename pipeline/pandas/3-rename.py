#!/usr/bin/env python3
"""
Renames a column and converts its values to datetime format
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime, converts values to datetime,
    and returns only Datetime and Close columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
