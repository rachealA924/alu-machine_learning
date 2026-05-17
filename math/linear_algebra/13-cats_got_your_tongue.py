#!/usr/bin/env python3
"""Concatenate numpy matrices"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
