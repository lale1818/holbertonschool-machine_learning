#!/usr/bin/env python3
"""Defines a function to compute descriptive statistics of a DataFrame"""
import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.
    """
    # Timestamp sütunu çıxmaqla statistika hesablayırıq
    return df.drop(columns=['Timestamp'], errors='ignore').describe()
