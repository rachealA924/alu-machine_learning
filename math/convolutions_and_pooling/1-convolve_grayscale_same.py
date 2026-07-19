#!/usr/bin/env python3
"""Same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
        for the convolution

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
