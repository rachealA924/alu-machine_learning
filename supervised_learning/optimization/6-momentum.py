#!/usr/bin/env python3
"""Create a TensorFlow 1.x momentum training operation."""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Returns the momentum optimizer operation."""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
