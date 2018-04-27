#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


sess = tf.InteractiveSession()
# Launch the graph

# a = tf.placeholder("float")
# b = tf.placeholder("float")
#
#
# v = tf.Variable(0)
#
# y = tf.pow(a, b)
# z = tf.add(y, 10)
# u = tf.add(z, 5)
# k = tf.add(u, v)
# print(sess.run(k.eval(), feed_dict={a: 2, b: 2}))

x = tf.placeholder(tf.float32)
f = 1 + 2 * x + tf.pow(x, 2)
sess.run(f, feed_dict={x: 10})

summary_writer = tf.summary.FileWriter('./', graph_def=sess.graph_def)