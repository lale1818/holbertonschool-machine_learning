#!/usr/bin/env python3
"""
Defines concat function
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes after indexing and filtering
    """
    df1 = index(df1)
    df2 = index(df2)

    df2_filtered = df2.loc[:1417411920]

    return pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
