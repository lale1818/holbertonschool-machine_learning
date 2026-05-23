#!/usr/bin/env python3
"""
Converts the last 10 rows of High and Close columns to a numpy array.
"""
import pandas as pd


def array(df):
    """ Selects last 10 rows of High and Close, returns numpy.ndarray """
    return df[['High', 'Close']].tail(10).to_numpy()
