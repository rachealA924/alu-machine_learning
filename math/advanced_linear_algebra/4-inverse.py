#!/usr/bin/env python3
"""Module for calculating the inverse matrix."""

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Calculates the inverse of a matrix"""

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    adj = adjugate(matrix)

    return [[element / det for element in row] for row in adj]
