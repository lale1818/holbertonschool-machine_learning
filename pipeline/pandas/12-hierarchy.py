#!/usr/bin/env python3
"""
Concatenates and rearranges MultiIndex for DataFrames
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ Concatenates and rearranges MultiIndex to put Timestamp first """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    start_ts, end_ts = 1417411980, 1417417980
    df1_filtered = df1_indexed.loc[start_ts:end_ts]
    df2_filtered = df2_indexed.loc[start_ts:end_ts]

    df_concat = pd.concat([df1_filtered, df2_filtered], keys=['coinbase', 'bitstamp'])
    df_concat = df_concat.swaplevel(0, 1)
    df_concat.sort_index(inplace=True)

    return df_concat
