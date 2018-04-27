#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

with tf.name_scope("TEST") as scope:
    x = tf.placeholder(tf.float32)
    f = 1 + 2 * x + tf.pow(x, 2)

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    sess.run(f, feed_dict={x: 10})
    summary_writer = tf.summary.FileWriter('./', graph_def=sess.graph_def)

    sess.run(f, feed_dict={x: 10})