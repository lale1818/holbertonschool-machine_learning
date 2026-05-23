#!/usr/bin/env python3
"""
Defines flip_switch function
"""


def flip_switch(df):
    """
    Sorts dataframe in reverse chronological order and transposes it
    """
    return df.iloc[::-1].T
