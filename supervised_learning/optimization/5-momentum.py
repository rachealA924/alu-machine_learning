#!/usr/bin/env python3
"""Update a variable using gradient descent with momentum."""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Returns the updated variable and its new first moment."""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
