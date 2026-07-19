#!/usr/bin/env python3
"""Convolution using multiple kernels on images"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images
    kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the
        kernels for the convolution
    padding is either a tuple of (ph, pw), 'same', or 'valid'
    stride is a tuple of (sh, sw)

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        ph = (ph + 1) // 2
        pw = max((w - 1) * sw + kw - w, 0)
        pw = (pw + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = padded[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            for k in range(nc):
                kernel = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(
                    image_slice * kernel, axis=(1, 2, 3)
                )

    return convolved
