#!/usr/bin/env python3
"""Data cleaning and visualization pipeline script"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove Weighted_Price column
df = df.drop(columns=['Weighted_Price'], errors='ignore')

# 2. Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Index the dataframe on Date
df = df.set_index('Date')

# 5. Missing values in Close set to previous row value
df['Close'] = df['Close'].ffill()

# 6. Missing values in High, Low, Open set to same row's Close value
df['Open'] = df['Open'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])

# 7. Missing values in Volume columns set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 8. Filter data from 2017 and beyond, and resample at daily intervals
df = df.loc['2017':]

df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plotting the data
df.plot()
plt.show()
