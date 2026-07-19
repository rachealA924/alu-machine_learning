#!/usr/bin/env python3
"""Strided convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution
    padding is either a tuple of (ph, pw), 'same', or 'valid'
    stride is a tuple of (sh, sw)

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
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
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = padded[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw
            ]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved