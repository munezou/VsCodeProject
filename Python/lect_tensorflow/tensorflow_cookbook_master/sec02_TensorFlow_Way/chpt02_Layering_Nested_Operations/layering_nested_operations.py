'''
-------------------------------------------------------------
Layering Nested Operations
-------------------------------------------------------------
'''
from __future__ import absolute_import, division, print_function, unicode_literals
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
def my_function():
    # Create the data and variables
    my_array = np.array(
                    [[1., 3., 5., 7., 9.],
                    [-2., 0., 2., 4., 6.],
                    [-6., -3., 0., 3., 6.]]
                    )
    x_vals = np.array([my_array, my_array + 1])

    # Constants for matrix multiplication:
    m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
    m2 = tf.constant([[2.]])
    a1 = tf.constant([[10.]])

    addition_result = []
    # Now feed data through placeholder and print results
    for x_val in x_vals:
        x_val = tf.dtypes.cast(x_val, tf.float32)
        
        # Create our multiple operations
        prod1 = tf.matmul(x_val, m1)
        prod2 = tf.matmul(prod1, m2)
        add1 = tf.add(prod2, a1)
        addition_result.append(add1)
    
    return addition_result

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, 'logs', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

calc_result = my_function()
for tmp in calc_result:
    print('calcu_result = \n{0}'.format(tmp))

with writer.as_default():
    tf.summary.trace_export(
        name = "Simple_fun",
        step = 0,
        profiler_outdir=logdir
    )

tf.summary.trace_off()