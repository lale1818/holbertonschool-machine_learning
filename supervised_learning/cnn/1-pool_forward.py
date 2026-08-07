#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Parameters:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - kernel_shape: tuple of (kh, kw)
    - stride: tuple of (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1

    A = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            i_st = i * sh
            j_st = j * sw
            slc_h = slice(i_st, i_st + kh)
            slc_w = slice(j_st, j_st + kw)
            img_slice = A_prev[:, slc_h, slc_w, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return A
