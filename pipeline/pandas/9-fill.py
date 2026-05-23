#!/usr/bin/env python3
"""
Defines fill function
"""


def fill(df):
    """
    Fills missing values in the dataframe according to specific rules
    """
    # Weighted_Price sütununu silirik
    df = df.drop(columns=['Weighted_Price'])

    # Close sütunundakı boşluqları əvvəlki sətrin dəyəri ilə doldururuq
    df['Close'] = df['Close'].ffill()

    # High, Low və Open sütunlarındakı boşluqları həmin sətrin Close dəyəri ilə doldururuq
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # Həcm sütunlarındakı boşluqları 0 ilə əvəzləyirik
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
