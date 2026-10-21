#!/usr/bin/env python3
"""Calculate a bias-corrected weighted moving average."""


def moving_average(data, beta):
    """Returns the bias-corrected moving averages."""
    averages = []
    v = 0.0
    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        averages.append(v / (1 - beta ** t))
    return averages
