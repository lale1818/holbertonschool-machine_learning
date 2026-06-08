#!/usr/bin/env python3
"""Defines a function to rearrange MultiIndex levels chronologically"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates and rearranges the MultiIndex structure chronologically.
    """
    # Hər iki DataFrame-i Timestamp sütununa görə indeksləyirik
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Müvafiq zaman aralığını inclusive (daxil olmaqla) filtrləyirik
    df1_filtered = df1_indexed.loc[1417411980:1417417980]
    df2_filtered = df2_indexed.loc[1417411980:1417417980]

    # Birləşdiririk
    result = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

    # İndeks səviyyələrinin yerini dəyişirik: (keys, Timestamp) -> (Timestamp, keys)
    result = result.swaplevel(0, 1)

    # Xronoloji olaraq indeksləri sıralayırıq
    result = result.sort_index()

    return result
