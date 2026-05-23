#!/usr/bin/env python3
"""
Defines fill function
"""


def fill(df):
    """
    Fills missing values in the dataframe according to specific rules
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    # Sətirlərin uzunluğu 79 simvoldan az saxlanıldı
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
