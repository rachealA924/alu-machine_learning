#!/usr/bin/env python3
"""Apply batch normalization to an unactivated layer output."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Returns the batch-normalized Z matrix."""
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
