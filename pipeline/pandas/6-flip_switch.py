#!/usr/bin/env python3
"""
Defines flip_switch function
"""
import pandas as pd


def flip_switch(df):
    """
    Sorts dataframe in reverse chronological order and transposes it
    """
    # .iloc[::-1] sətirləri tərs çevirir, .T isə transpose edir
    return df.iloc[::-1].T
