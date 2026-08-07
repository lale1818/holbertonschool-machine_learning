#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.

    Parameters:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
    - padding: "same" or "valid"
    - stride: tuple of (sh, sw)

    Returns:
    - dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    padded_A = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_pad = np.zeros_like(padded_A)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                i_st = i * sh
                j_st = j * sw
                slc_h = slice(i_st, i_st + kh)
                slc_w = slice(j_st, j_st + kw)

                img_slice = padded_A[:, slc_h, slc_w, :]
                dz_slice = dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis]

                dW[:, :, :, k] += np.sum(
                    img_slice * dz_slice, axis=0
                )
                dA_pad[:, slc_h, slc_w, :] += (
                    W[:, :, :, k] * dz_slice
                )

    if ph == 0 and pw == 0:
        dA_prev = dA_pad
    else:
        dA_prev = dA_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
