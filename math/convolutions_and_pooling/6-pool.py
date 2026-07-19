#!/usr/bin/env python3
"""Pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
        images
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for
        the pooling
    stride is a tuple of (sh, sw)
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            image_slice = images[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
            ]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
