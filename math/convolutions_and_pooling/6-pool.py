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
    - numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            i_st = i * sh
            j_st = j * sw
            slice_h = slice(i_st, i_st + kh)
            slice_w = slice(j_st, j_st + kw)
            img_slice = images[:, slice_h, slice_w, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return pooled
