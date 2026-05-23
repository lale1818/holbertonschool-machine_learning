#!/usr/bin/env python3
"""
Fills missing values in the cryptocurrency dataset.
"""
import pandas as pd


def fill(df):
    """
    Fills missing values according to task specifications.
    Modifies the DataFrame in place and returns it.
    """
    df.drop(columns=['Weighted_Price'], inplace=True)
    
    df['Close'] = df['Close'].ffill()
    
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    
    return df
