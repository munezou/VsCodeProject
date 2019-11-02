import tensorflow as tf

import cProfile

tf.executing_eagerly()
print('tf.executing_eagerly = {0}'.format(tf.executing_eagerly()))
print()


x = [[2.]]
m = tf.matmul(x, x)
print('hello = {0}'.format(m))
print()

@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

print('simple_nn_layer(x, y) = \n{0}'.format(simple_nn_layer(x, y)))
print()

