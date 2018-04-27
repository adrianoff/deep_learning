#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!') #создаем объект из TF
sess = tf.InteractiveSession() #создаем сессию
print(sess.run(hello)) #сессия "выполняет" объект


zeros_tensor = tf.zeros([3, 3])
print(zeros_tensor.eval())
print(zeros_tensor)


a = tf.truncated_normal([2, 2])
b = tf.fill([2, 2], 0.5)
print(sess.run(a + b))
print(sess.run(a - b))
print(sess.run(a * b))
print(sess.run(tf.matmul(a, b)))


v = tf.Variable(zeros_tensor)
sess.run(v.initializer)
print (v.eval())



x = tf.placeholder(tf.float32, shape=(4, 4))
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.pow(a, b)
print(sess.run(y, feed_dict={a:2, b:2}))



with tf.name_scope("test") as scope:
    x = tf.placeholder(tf.float32)
    f = 1 + 2 * x + tf.pow(x, 2)
    print(sess.run(f, feed_dict={x: 10}))