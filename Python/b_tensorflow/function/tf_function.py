# tensorflow 2.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
-----------------------------------------------------------------------------------------------
Better performance with tf.function and AutoGraph

TF 2.0 brings together the ease of eager execution and the power of TF 1.0. 
At the center of this merger is tf.function, which allows you to transform a subset of Python syntax into portable, 
high-performance TensorFlow graphs.

A cool new feature of tf.function is AutoGraph, which lets you write graph code using natural Python syntax. 
For a list of the Python features that you can use with AutoGraph, see AutoGraph Capabilities and Limitations. 
For more details about tf.function, see the RFC TF 2.0: 
Functions, not Sessions. For more details about AutoGraph, see tf.autograph.

This tutorial will walk you through the basic features of tf.function and AutoGraph.
-----------------------------------------------------------------------------------------------
'''

print(__doc__)

# common library
import os
import platform
import shutil
import subprocess
from packaging import version
import timeit

import numpy as np
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The tf.function decorator                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
When you annotate a function with tf.function, you can still call it like any other function. 
But it will be compiled into a graph, which means you get the benefits of faster execution, 
running on GPU or TPU, or exporting to SavedModel.
---------------------------------------------------------------------------------------------------------------
'''
@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer_perform = simple_nn_layer(x, y)
print()
print('simple_nn_layer_perform = \n{0}\n'.format(simple_nn_layer_perform))

'''
----------------------------------------------------------------------------------------------------------------
If we examine the result of the annotation, 
we can see that it's a special callable that handles all interactions with the TensorFlow runtime.
----------------------------------------------------------------------------------------------------------------
'''
print('simple_nn_layer = \n{0}\n'.format(simple_nn_layer))

'''
----------------------------------------------------------------------------------------------------------------
If your code uses multiple functions, 
you don't need to annotate them all - any functions called from an annotated function will also run in graph mode.
----------------------------------------------------------------------------------------------------------------
'''
def linear_layer(x):
    return 2 * x + 1


@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

deep_net_performance = deep_net(tf.constant((1, 2, 3)))
print('deep_net_perform = {0}\n'.format(deep_net_performance))
print('deep_net = \n{0}\n'.format(deep_net))

'''
---------------------------------------------------------------------------------------------------------------
Functions can be faster than eager code, 
for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), 
you may not see much speedup.
---------------------------------------------------------------------------------------------------------------
'''
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
    return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])

# warm up
conv_layer(image); conv_fn(image)
print("Eager conv: {0}\n".format(timeit.timeit(lambda: conv_layer(image), number=10)))
print("Function conv: {0}\n".format(timeit.timeit(lambda: conv_fn(image), number=10)))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
    return lstm_cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2

# warm up
lstm_cell(input, state); lstm_fn(input, state)
print("eager lstm: {0}\n".format(timeit.timeit(lambda: lstm_cell(input, state), number=10)))
print("function lstm: {0}\n".format(timeit.timeit(lambda: lstm_fn(input, state), number=10)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Use Python control flow                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
When using data-dependent control flow inside tf.function, 
you can use Python control flow statements and AutoGraph will convert them into appropriate TensorFlow ops. 
For example, if statements will be converted into tf.cond() if they depend on a Tensor.
---------------------------------------------------------------------------------------------------------------
'''
# In the example below, x is a Tensor but the if statement works as expected:
