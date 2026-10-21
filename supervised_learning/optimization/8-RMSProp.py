#!/usr/bin/env python3
"""Create a TensorFlow 1.x RMSProp training operation."""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Returns the RMSProp optimizer operation."""
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
