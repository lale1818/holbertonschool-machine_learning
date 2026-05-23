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
    # bitstamp-dan 1417411920 daxil olmaqla ondan əvvəlki sətirləri seçirik
    df2_filtered = df2[df2['Timestamp'] <= 1417411920]

    # İndi hər iki DataFrame üçün Timestamp sütununu indeks təyin edirik
    df1 = index(df1)
    df2_filtered = index(df2_filtered)

    # İndekslənmiş məlumatları coinbase üstə, bitstamp alta gələcək şəkildə yığırıq
    return pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
