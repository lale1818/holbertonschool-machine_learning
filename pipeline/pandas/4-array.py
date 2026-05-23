#!/usr/bin/env python3
"""
Defines array function
"""
import pandas as pd


def array(df):
    """
    Converts columns to ndarray
    """
    return df[['High', 'Close']].tail(10).to_numpy()
