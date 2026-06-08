#!/usr/bin/env python3
"""
Module that removes rows with missing Close values
"""

import pandas as pd


def prune(df):
    """Removes rows where Close is NaN"""
    return df.dropna(subset=['Close'])
