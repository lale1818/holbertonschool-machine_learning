#!/usr/bin/env python3
"""
Defines a function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    Columns are labeled in alphabetical order and capitalized
    """
    num_cols = array.shape[1]
    col_names = [chr(65 + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=col_names)
