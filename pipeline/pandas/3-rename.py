#!/usr/bin/env python3
"""
Defines a function that renames and modifies a pd.DataFrame column
"""
import pandas as pd


def rename(df):
    """
    Renames Timestamp to Datetime, converts it to datetime values,
    and returns only Datetime and Close columns
    """
    # 1. Sütunun adını dəyişirik
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # 2. Saniyə (timestamp) dəyərlərini oxunabilən tarix formatına çeviririk
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # 3. Ancaq Datetime və Close sütunlarını seçib qaytarırıq
    return df[['Datetime', 'Close']]
