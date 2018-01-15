import tensorflow as tf

test = tf.reduce_mean([[1, 1, 1], [5, 5, 8], [3, 3, 3]], 0)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print sess.run(test)