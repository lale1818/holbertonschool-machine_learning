#!/usr/bin/env python3
"""
Defines a function that converts parts of a pd.DataFrame into a numpy.ndarray
"""
import pandas as pd


def array(df):
    """
    Selects the last 10 rows of High and Close columns
    and converts them into a numpy.ndarray
    """
    # High və Close sütunlarının son 10 sətrini seçib .to_numpy() ilə massivə çeviririk
    return df[['High', 'Close']].tail(10).to_numpy()
