#!/usr/bin/env python3
"""
Plots a 3D scatter plot of the PCA-reduced Iris dataset.
"""
import matplotlib.pyplot as plt
import numpy as np


def pca_3d():
    """ Visualizes 3D PCA components of Iris dataset using plasma cmap """
    lib = np.load("pca.npz")
    data = lib["data"]
    labels = lib["labels"]
    data_means = np.mean(data, axis=0)
    norm_data = data - data_means
    _, _, Vh = np.linalg.svd(norm_data)
    pca_data = np.matmul(norm_data, Vh[:3].T)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
               c=labels, cmap='plasma')
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('U3')
    ax.set_title('PCA of Iris Dataset')
    plt.show()


if __name__ == '__main__':
    pca_3d()
