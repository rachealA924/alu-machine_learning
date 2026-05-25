#!/usr/bin/env python3
"""Module for calculating the minor matrix."""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix"""

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []

    for i in range(n):
        row = []

        for j in range(n):
            submatrix = [
                r[:j] + r[j + 1:]
                for k, r in enumerate(matrix) if k != i
            ]

            row.append(determinant(submatrix))

        minor_matrix.append(row)

    return minor_matrix
    