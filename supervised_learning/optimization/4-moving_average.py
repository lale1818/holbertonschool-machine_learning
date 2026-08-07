#!/usr/bin/env python3
"""
This module contains a function that calculates the exponentially weighted
moving average of a data set with bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Parameters:
    - data: list of data points to calculate the moving average of
    - beta: weight used for the moving average

    Returns:
    - a list containing the moving averages of data
    """
    v = 0
    moving_averages = []

    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - (beta ** i))
        moving_averages.append(v_corrected)

    return moving_averages
