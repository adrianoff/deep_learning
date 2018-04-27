#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.Session()
#sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.sigmoid(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
training = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

SIZE_TRAIN = 10000
SIZE_TEST = 5000
x_train = mnist.train.images[:SIZE_TRAIN, :]
y_train = mnist.train.labels[:SIZE_TRAIN, :]
x_test = mnist.train.images[:SIZE_TEST, :]
y_test = mnist.train.labels[:SIZE_TEST, :]

sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
sess.run(training, feed_dict={x: x_train, y_: y_train})
print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
