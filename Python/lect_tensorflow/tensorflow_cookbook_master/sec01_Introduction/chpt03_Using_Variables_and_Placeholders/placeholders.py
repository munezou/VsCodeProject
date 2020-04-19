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

@tf.function
def simple_func(arg):
    a = tf.constant(7.9)
    b = tf.constant(6.3)
    c = arg + a
    d = a * b
    ret = c + d

    return ret

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, 'log', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing.
arg = tf.constant(8.9)
print(simple_func(arg))

with writer.as_default():
    tf.summary.trace_export(
        name = "Simple_fun",
        step = 0,
        profiler_outdir=logdir
    )

tf.summary.trace_off()