#!/usr/bin/env python3
"""Create a TensorFlow 1.x Adam training operation."""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Returns the Adam optimizer operation."""
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
