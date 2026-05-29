#!/usr/bin/env python3
"""
Plots a scatter plot of sampled elevations on a mountain with a colorbar.
"""
import matplotlib.pyplot as plt
import numpy as np


def gradient():
    """ Plots x, y coordinates with z as color elevation and a colorbar """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    scatter_plot = plt.scatter(x, y, c=z)
    
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    
    cbar = plt.colorbar(scatter_plot)
    cbar.set_label('elevation (m)')
    
    plt.show()
