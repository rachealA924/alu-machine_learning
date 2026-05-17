#!/usr/bin/env python3
"""Matrix shape"""


def matrix_shape(matrix):
    """Returns shape of matrix"""
    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape