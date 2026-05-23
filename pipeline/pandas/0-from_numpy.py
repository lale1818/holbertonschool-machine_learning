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
    # Sütunların sayını tapırıq
    num_cols = array.shape[1]

    # Sütun adları üçün 'A'-dan başlayaraq lazımi sayda hərf generatsiya edirik
    # ASCII 65 = 'A' hərfi. Heç bir əlavə modul import etmədən chr() istifadə edirik.
    col_names = [chr(65 + i) for i in range(num_cols)]

    # DataFrame yaradıb geri qaytarırıq
    return pd.DataFrame(array, columns=col_names)
