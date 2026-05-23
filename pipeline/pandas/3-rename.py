#!/usr/bin/env python3
"""
Modifies a DataFrame by renaming columns and converting time formats.
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts it to datetime objects,
    and modifies the DataFrame to keep only Datetime and Close columns.
    """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    
    columns_to_drop = [col for col in df.columns if col not in ['Datetime', 'Close']]
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df
