#!/usr/bin/env python3
"""
Module to create a DataFrame from a Numpy array
"""
import pandas as pd
import string


def from_numpy(array):
    """ Creates a pd.DataFrame from a np.ndarray """
    num_cols = array.shape[1]
    col_names = list(string.ascii_uppercase[:num_cols])
    return pd.DataFrame(array, columns=col_names)
