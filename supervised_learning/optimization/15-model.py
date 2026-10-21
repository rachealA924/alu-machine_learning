#!/usr/bin/env python3
"""Build and train a TensorFlow 1.x network with batch norm and Adam."""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Build, train, evaluate, and save the neural network."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name="x")
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name="y")

    A = x
    for n, activation in zip(layers, activations):
        A = create_batch_norm_layer(A, n, activation)
    y_pred = tf.nn.softmax(A, name="y_pred")
    loss = tf.reduce_mean(
        -tf.reduce_sum(y * tf.log(y_pred + 1e-8), axis=1), name="loss")
    correct = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    global_step = tf.Variable(0, trainable=False, name="epoch_step")
    learning_rate = learning_rate_decay(alpha, decay_rate, global_step, 1)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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
            sess.run(global_step.assign_add(1))
            report(epoch + 1)

        saved_path = saver.save(sess, save_path)
    return saved_path
