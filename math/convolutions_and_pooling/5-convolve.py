#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c)
    - kernels: numpy.ndarray of shape (kh, kw, c, nc)
    - padding: 'same', 'valid', or tuple (ph, pw)
    - stride: tuple (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                i_st = i * sh
                j_st = j * sw
                slc_h = slice(i_st, i_st + kh)
                slc_w = slice(j_st, j_st + kw)
                img_slice = padded[:, slc_h, slc_w, :]
                k_slice = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(
                    img_slice * k_slice, axis=(1, 2, 3)
                )

    return convolved
