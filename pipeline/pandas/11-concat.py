#!/usr/bin/env python3
"""Defines a function to concatenate two DataFrames with specific indexing and keys"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates specific rows of df2 to the top of df1 after indexing.

    Args:
        df1 (pd.DataFrame): Coinbase data.
        df2 (pd.DataFrame): Bitstamp data.

    Returns:
        pd.DataFrame: The concatenated dataframe.
    """
    # Hər iki DataFrame-i Timestamp sütununa görə indeksləyirik
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # df2-ni timestamp 1417411920-yə qədər (daxil olmaqla) filtrləyirik
    df2_filtered = df2_indexed.loc[:1417411920]

    # df2_filtered hissəsini df1-in üstünə birləşdiririk və key-ləri veririk
    result = pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])

    return result
