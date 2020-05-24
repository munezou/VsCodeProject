'''
-----------------------------------
Working with Multiple Layers
-----------------------------------
'''
# First we start with loading the necessary libraries and resetting the computational graph.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import io
import datetime
from packaging import version
import numpy as np
import tensorflow as tf

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

@tf.function
def first_layer():
    # Create tensors
    # Create a small random 'image' of size 4x4
    x_shape = [1, 4, 4, 1]
    x_val = np.random.uniform(size=x_shape)

    # Create a layer that takes a spatial moving window average
    # Our window will be 2x2 with a stride of 2 for height and width
    # The filter value will be 0.25 because we want the average of the 2x2 window
    my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    
    x_val = tf.dtypes.cast(x_val, tf.float32)
    mov_avg_layer= tf.nn.conv2d(x_val, my_filter, my_strides,
                        padding='SAME', name='Moving_Avg_Window')
    
    return mov_avg_layer

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir1 = os.path.join(PROJECT_ROOT_DIR, 'logs', 'first_layer', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir1)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

calc_result = first_layer()
print('calcu_result = \n{0}'.format(calc_result))

with writer.as_default():
    tf.summary.trace_export(
        name = "first_layer",
        step = 0,
        profiler_outdir=logdir1
    )

tf.summary.trace_off()

@tf.function
def second_layer():
    # Create tensors
    # Create a small random 'image' of size 4x4
    x_shape = [1, 4, 4, 1]
    x_val = np.random.uniform(size=x_shape)

    # Create a layer that takes a spatial moving window average
    # Our window will be 2x2 with a stride of 2 for height and width
    # The filter value will be 0.25 because we want the average of the 2x2 window
    my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    
    x_val = tf.dtypes.cast(x_val, tf.float32)
    mov_avg_layer= tf.nn.conv2d(x_val, my_filter, my_strides,
                        padding='SAME', name='Moving_Avg_Window')
    
    input_matrix_sqeezed = tf.squeeze(mov_avg_layer)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.linalg.matmul(A, input_matrix_sqeezed)
    temp = tf.math.add(temp1, b)
    custom_layer1 = tf.sigmoid(temp)
    return custom_layer1

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir2 = os.path.join(PROJECT_ROOT_DIR, 'logs', '2nd_layer', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir2)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

calc_result = second_layer()
print('calcu_result = \n{0}'.format(calc_result))

with writer.as_default():
    tf.summary.trace_export(
        name = "first_layer",
        step = 0,
        profiler_outdir=logdir2
    )

tf.summary.trace_off()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         multiple_layers.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()