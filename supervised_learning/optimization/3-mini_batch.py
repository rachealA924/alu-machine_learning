#!/usr/bin/env python3
"""Train a restored TensorFlow 1.x graph using mini-batch gradient descent."""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Train a restored model using mini-batches and save it."""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        graph = tf.get_default_graph()
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        def report(epoch):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        report(0)
        for epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            step = 0
            for start in range(0, X_shuffled.shape[0], batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                Y_batch = Y_shuffled[start:start + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step += 1
                if step % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
            report(epoch + 1)

        saved_path = saver.save(sess, save_path)
    return saved_path
