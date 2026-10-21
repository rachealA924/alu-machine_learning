#!/usr/bin/env python3
"""Create a TensorFlow 1.x dense layer with batch normalization."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Returns the activated output of a batch-normalized dense layer."""
    dense = tf.layers.Dense(
        n,
        activation=None,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"))
    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), name="gamma")
    beta = tf.Variable(tf.zeros([n]), name="beta")
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = (Z - mean) / tf.sqrt(variance + 1e-8)
    Z_scaled = gamma * Z_norm + beta
    if activation is not None:
        return activation(Z_scaled)
    return Z_scaled
