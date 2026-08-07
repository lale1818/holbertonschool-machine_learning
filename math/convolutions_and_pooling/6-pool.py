#!/usr/bin/env python3
"""
Module to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c)
    - kernel_shape: tuple of (kh, kw)
    - stride: tuple of (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - numpy.ndarray containing the pooled images of shape (m, out_h, out_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            image_slice = images[:, i_start:i_start + kh, j_start:j_start + kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
