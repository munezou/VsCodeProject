'''
-----------------------------------------------------------------
Placeholders

This function introduces how to use placeholders in TensorFlow
-----------------------------------------------------------------
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import datetime
from packaging import version
import numpy as np
import tensorflow as tf

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

# The function to be traced
@tf.function
def my_func():
    x = tf.Variable(np.random.rand(4, 4))
    y = tf.identity(x)
    return y

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, 'log/{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing.
z = my_func()

with writer.as_default:
    tf.summary.trace_export(
        name = "my_func_trace",
        step = 0,
        profiler_outdir=logdir
    )