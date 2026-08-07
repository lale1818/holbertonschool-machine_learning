#!/usr/bin/env python3
"""
Module to perform back propagation over a pooling layer.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    Parameters:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
    - kernel_shape: tuple of (kh, kw)
    - stride: tuple of (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - dA_prev: partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c):
                i_st = i * sh
                j_st = j * sw
                slc_h = slice(i_st, i_st + kh)
                slc_w = slice(j_st, j_st + kw)

                if mode == 'max':
                    for ex in range(m):
                        a_slice = A_prev[ex, slc_h, slc_w, k]
                        mask = (a_slice == np.max(a_slice))
                        da = dA[ex, i, j, k]
                        dA_prev[ex, slc_h, slc_w, k] += mask * da

                elif mode == 'avg':
                    da = dA[:, i, j, k, np.newaxis, np.newaxis]
                    avg_da = da / (kh * kw)
                    dA_prev[:, slc_h, slc_w, k] += avg_da

    return dA_prev
