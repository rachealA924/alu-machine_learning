#!/usr/bin/env python3

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Calculates the adjugate matrix"""

    cof = cofactor(matrix)
    n = len(cof)

    return [[cof[j][i] for j in range(n)] for i in range(n)]