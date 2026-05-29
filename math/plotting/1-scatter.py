#!/usr/bin/env python3
"""
Plots a scatter plot of height vs weight
"""
import matplotlib.pyplot as plt
import numpy as np


def scatter():
    """ Plots height vs weight as magenta points with proper labels """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, color='m')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
