#!/usr/bin/env python3
"""Normalize a matrix using supplied feature means and standard deviations."""
import numpy as np


def normalize(X, m, s):
    """Returns the normalized X matrix."""
    return (X - m) / s
