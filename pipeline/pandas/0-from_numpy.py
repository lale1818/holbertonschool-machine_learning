#!/usr/bin/env python3
"""Defines a function that creates a DataFrame from a NumPy array"""
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray with alphabetical column labels"""
    # Sütun sayını alırıq
    num_cols = array.shape[1]
    
    # Sütun sayına uyğun əlifba hərflərini (A, B, C...) generatsiya edirik
    columns = [chr(65 + i) for i in range(num_cols)]
    
    # DataFrame yaradıb qaytarırıq
    return pd.DataFrame(array, columns=columns)
