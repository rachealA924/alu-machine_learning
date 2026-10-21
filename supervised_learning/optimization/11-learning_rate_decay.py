#!/usr/bin/env python3
"""Apply stepwise inverse-time learning-rate decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Returns the decayed learning rate."""
    return alpha / (1 + decay_rate * (global_step // decay_step))
