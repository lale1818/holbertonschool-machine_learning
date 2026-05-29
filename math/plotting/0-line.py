#!/usr/bin/env python3
"""
Plots y as a line graph
"""
import matplotlib.pyplot as plt
import numpy as np


def line():
    """ Plots y as a solid red line with x-axis ranging from 0 to 10 """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
