#!/usr/bin/env python3
"""Calculates the derivative of a polynomial."""


def poly_derivative(poly):
    """Returns the derivative of a polynomial."""
    if (not isinstance(poly, list) or
            len(poly) == 0 or
            not all(isinstance(x, (int, float)) for x in poly)):
        return None

    if len(poly) == 1:
        return [0]

    derivative = []

    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)

    if derivative == []:
        return [0]

    return derivative