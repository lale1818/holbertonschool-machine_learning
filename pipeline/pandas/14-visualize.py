#!/usr/bin/env python3
"""
Module that visualizes Bitcoin data
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ','
)

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp → Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set index
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].ffill()

df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter from 2017
df = df[df.index.year >= 2017]

# Resample daily
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot (NOT printed, just required)
df.plot()
plt.show()

# Return dataframe
print(df)
