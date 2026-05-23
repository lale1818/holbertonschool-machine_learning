#!/usr/bin/env python3
"""
Defines concat function
"""
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes after indexing and filtering
    """
    # Hər iki dataframe-i Timestamp sütununa görə indeksləyirik
    df1 = index(df1)
    df2 = index(df2)

    # df2 daxilindən 1417411920 zamanına qədər olan sətirləri seçirik
    df2_filtered = df2.loc[:1417411920]

    # concat funksiyası üçün pd-yə birbaşa df1.__class__ vasitəsilə müraciət
    pd = df1.__class__.concat

    # df2_filtered-i df1-in üzərinə əlavə edirik və açarları təyin edirik
    return pd([df2_filtered, df1], keys=['bitstamp', 'coinbase'])
