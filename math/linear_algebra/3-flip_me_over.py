#!/usr/bin/env python3
"""Transpose matrix"""


def matrix_transpose(matrix):
    """Returns transpose"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
