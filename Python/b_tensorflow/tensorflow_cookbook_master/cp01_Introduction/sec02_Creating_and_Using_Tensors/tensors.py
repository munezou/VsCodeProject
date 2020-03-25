from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
-----------------------------------------------------------------------------------------------
Tensors

This function introduces various ways to create
tensors in TensorFlow
tensorflow 2.0
-----------------------------------------------------------------------------------------------
'''
print(__doc__)

import os
import numpy as np
import math
from datetime import datetime

import tensorflow as tf

# Display current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

# Introduce tensors in tf
my_tensor = tf.zeros([1,20])
print('my_tensor = \n{0}\n'.format(my_tensor))

# Different kinds of variables
row_dim = 2
col_dim = 3 

# Zero initialized variable
zero_tensor = tf.zeros([row_dim, col_dim])
print('zero_tensor = \n{0}\n'.format(zero_tensor))

# One initialized variable
ones_tensor = tf.ones([row_dim, col_dim])
print('ones_tensor = \n{0}\n'.format(ones_tensor))

# shaped like other variable
zero_similar = tf.zeros_like(zero_tensor)
ones_similar = tf.ones_like(ones_tensor)
print('zero_similar = \n{0}\n'.format(zero_similar))
print('ones_similar = \n{0}\n'.format(ones_similar))

# Fill shape with a constant
fill_tensor = tf.fill([row_dim, col_dim], -1)
print('fill_tensor = \n{0}\n'.format(fill_tensor))

# Create a variable from a constant
const_tensor = tf.constant([8, 6, 7, 5, 3, 0, 9])
print('const_tensor = \n{0}\n'.format(const_tensor))

# This can also be used to fill an array:
const_fill_tensor = tf.constant(-1, shape=[row_dim, col_dim])
print('const_fill_tensor = \n{0}\n'.format(const_fill_tensor))

# Sequence generation
linear_tensor = tf.linspace(start=0.0, stop=1.0, num=3) # Generates [0.0, 0.5, 1.0] includes the end
print('linear_tensor = \n{0}\n'.format(linear_tensor))

# The function to be traced.
@tf.function
def my_func(x, y):
    # A simple hand-rolled layer.
    return tf.nn.relu(tf.matmul(x, y))

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, "logs", "func", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
linear_tensor = tf.linspace(start=0.0, stop=1.0, num=3)
z = z + linear_tensor
with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=3,
        profiler_outdir=logdir)


'''
# Random Numbers

# Random Normal
rnorm_tensor = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

# Add summaries to tensorboard
merged = tf.summary.merge_all()

# Initialize graph writer:
writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# Initialize operation
initialize_op = tf.global_variables_initializer()

# Run initialization of variable
sess.run(initialize_op)
'''