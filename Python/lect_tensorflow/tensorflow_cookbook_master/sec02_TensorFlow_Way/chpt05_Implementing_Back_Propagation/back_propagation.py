'''
*****************************************************************
# Back Propagation
#----------------------------------------------------------------
#
# This python function shows how to implement back propagation
# in regression and classification models.
****************************************************************
'''
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

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print(
    '-----------------------------------------------------------------------------------------\n'
    '   Regression Example:                                                                   \n'
    '   We will create sample data as follows:                                                \n'
    '   x-data: 100 random samples from a normal ~ N(1, 0.1)                                  \n'
    '   target: 100 values of the value 10.                                                   \n'
    '   We will fit the model:                                                                \n'
    '   x-data * A = target                                                                   \n'
    '   Theoretically, A = 10.                                                                \n'
    '-----------------------------------------------------------------------------------------\n'
)


# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

opt = tf.keras.optimizers.SGD(0.02)

result_A = []
result_dloss_da = []

@tf.function
def train(j, a, x, y):    
    # Add L2 loss operation to graph
    with tf.GradientTape() as tape:
        loss = tf.square(tf.math.multiply(x, a) - y)
    
    dloss_da = tape.gradient(loss, a)
    if (j+1)%25==0:
        tf.print('dloss_da =')
        tf.print(dloss_da)
        result_dloss_da.append(dloss_da)
    opt.apply_gradients([(dloss_da, a)])
    if (j+1)%25==0:
        tf.print('a =')
        tf.print(a)
        tf.print()
        result_A.append(a)

for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_x = tf.cast(rand_x, dtype=tf.float32)
    rand_y = [y_vals[rand_index]]
    rand_y = tf.cast(rand_y, dtype=tf.float32)

    # Create variable (one model parameter = A)
    A = tf.Variable(tf.random.normal(shape=[1]))
    A = tf.cast(A, dtype=tf.float32)

    print('A = {0}, x = {1}, y = {2}\n'.format(A, rand_x, rand_y))

    print('---< Convergence measurement >---')

    # Run Loop
    for j in range(200):
        train(j, A, rand_x, rand_y)




date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         back_propagation.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()