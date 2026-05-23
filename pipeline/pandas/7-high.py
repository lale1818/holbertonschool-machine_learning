#!/usr/bin/env python3
"""
Sorts a DataFrame by the High price in descending order.
"""
import pandas as pd


def high(df):
    """ Sorts the DataFrame by High column in descending order """
    return df.sort_values(by='High', ascending=False)
