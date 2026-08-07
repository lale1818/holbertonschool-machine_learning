#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
    - activation: activation function applied to the convolution
    - padding: "same" or "valid"
    - stride: tuple of (sh, sw)

    Returns:
    - output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_new):
                i_st = i * sh
                j_st = j * sw
                slc_h = slice(i_st, i_st + kh)
                slc_w = slice(j_st, j_st + kw)
                img_slice = padded[:, slc_h, slc_w, :]
                kernel = W[:, :, :, k]
                bias = b[:, :, :, k]
                Z[:, i, j, k] = (
                    np.sum(img_slice * kernel, axis=(1, 2, 3)) + bias
                )

    return activation(Z)
