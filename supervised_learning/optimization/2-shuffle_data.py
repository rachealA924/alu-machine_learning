#!/usr/bin/env python3
"""Shuffle two datasets using the same permutation."""
import numpy as np


def shuffle_data(X, Y):
    """Returns X and Y shuffled in the same order."""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
