'''
-----------------------------------------
Operations on a Computational Graph
-----------------------------------------
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
def simple_func():
    # Call only one tf.function when tracing.
    # Create data to feed in the placeholder
    x_vals = np.array([1., 3., 5., 7., 9.])
    # Constant for multilication
    m = tf.constant(3.)

    x_data = []
    for x_val in x_vals:
        x_val = tf.dtypes.cast(x_val, tf.float32)
        x_data.append(tf.math.multiply(x_val, m))

    return x_data

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(PROJECT_ROOT_DIR, 'logs', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

z = simple_func()
for data in z:
    print('data = {0}'.format(data))

with writer.as_default():
    tf.summary.trace_export(
        name = "Simple_fun",
        step = 0,
        profiler_outdir=logdir
    )

tf.summary.trace_off()