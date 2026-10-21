#!/usr/bin/env python3
"""Calculate the mean and standard deviation of each feature."""
import numpy as np


def normalization_constants(X):
    """Returns the mean and standard deviation of each feature."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
