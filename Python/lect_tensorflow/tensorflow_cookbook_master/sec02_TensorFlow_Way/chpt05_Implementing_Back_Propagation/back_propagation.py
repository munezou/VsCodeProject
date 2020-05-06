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

# Regression Example:
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

@tf.function
def train(a, x, y, opt):    
    # Add L2 loss operation to graph
    loss = tf.square(tf.math.multiply(x, a) - y)
    
    with tf.GradientTape() as tape:
        dv = tape.gradient(loss, [loss])
    
    opt.apply_gradients(zip(dv, [loss]))
    print(dv[0], loss)

# Run Loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_x = tf.cast(rand_x, dtype=tf.float32)
    rand_y = [y_vals[rand_index]]
    rand_y = tf.cast(rand_y, dtype=tf.float32)
    
    # Create variable (one model parameter = A)
    A = tf.Variable(tf.random.normal(shape=[1]))
    A = tf.cast(A, dtype=tf.float32)
    
    train(A, rand_x, rand_y, tf.keras.optimizers.SGD(0.02))