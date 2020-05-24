'''
********************************************************
# Loss Functions
#----------------------------------
#
#  This python script illustrates the different
#  loss functions for regression and classification.
********************************************************
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import io
import datetime
from packaging import version
import matplotlib.pyplot as plt
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

print()

###### Numerical Predictions ######
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           L2 Loss                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The L2 loss is one of the most common regression loss functions. 
Here we show how to create it in TensorFlow and we evaluate it for plotting later.
---------------------------------------------------------------------------------------------------------------
'''
# L = (pred - actual)^2
l2_y_vals = tf.square(target - x_vals)
print('l2_y_vals = \n{0}\n'.format(l2_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           L1 Loss                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
An alternative loss function to consider is the L1 loss. 
This is very similar to L2 except that we take the absolute value of the difference instead of squaring it.
---------------------------------------------------------------------------------------------------------------
'''
# L = abs(pred - actual)
l1_y_vals = tf.abs(target - x_vals)
print('l1_y_vals = \n{0}\n'.format(l1_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Pseudo-Huber loss                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The psuedo-huber loss function is a smooth approximation to the L1 loss as the (predicted - target) values get larger. 
When the predicted values are close to the target, the pseudo-huber loss behaves similar to the L2 loss.
---------------------------------------------------------------------------------------------------------------
'''
# L = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
print('phuber1_y_vals = \n{0}\n'.format(phuber1_y_vals))

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
print('phuber2_y_vals = \n{0}\n'.format(phuber2_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Plot the Regression Losses                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Here we use Matplotlib to plot the L1, L2, and Pseudo-Huber Losses.
---------------------------------------------------------------------------------------------------------------
'''
# Plot the output:
x_array = x_vals
plt.plot(x_array, l2_y_vals, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_vals, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_vals, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_vals, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.grid()
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Categorical Predictions                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
We now consider categorical loss functions. 
Here, the predictions will be around the target of 1.
--------------------------------------------------------------------------------------------------------------
'''
# Various predicted X values
x_vals = tf.linspace(-3., 5., 500)

# Target of 1.0
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Hinge Loss                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
The hinge loss is useful for categorical predictions. 
Here is the max(0, 1-(pred*actual)).
--------------------------------------------------------------------------------------------------------------
'''

# Use for predicting binary (-1, 1) classes
# L = max(0, 1 - (pred * actual))
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
print('hinge_y_vals = \n{0}\n'.format(hinge_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Cross Entropy Loss                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
The cross entropy loss is a very popular way to measure the loss between categorical targets and output model logits. 
You can read about the details more here: https://en.wikipedia.org/wiki/Cross_entropy
--------------------------------------------------------------------------------------------------------------
'''
# Cross entropy loss
# L = -actual * (log(pred)) - (1-actual)(log(1-pred))
xentropy_y_vals = - tf.math.multiply(target, tf.math.log(x_vals)) - tf.multiply((1. - target), tf.math.log(1. - x_vals))
print('xentropy_y_vals = \n{0}\n'.format(xentropy_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Sigmoid Entropy Loss                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
TensorFlow also has a sigmoid-entropy loss function. 
This is very similar to the above cross-entropy function except that we take the sigmoid of the predictions in the function.
---------------------------------------------------------------------------------------------------------------
'''
# L = -actual * (log(sigmoid(pred))) - (1-actual)(log(1-sigmoid(pred)))
# or
# L = max(actual, 0) - actual * pred + log(1 + exp(-abs(actual)))
x_val_input = tf.expand_dims(x_vals, 1)
target_input = tf.expand_dims(targets, 1)
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_val_input, labels=target_input)
print('xentropy_sigmoid_y_vals = \n{0}\n'.format(xentropy_sigmoid_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Weighted (Softmax) Cross Entropy Loss                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Tensorflow also has a similar function to the sigmoid cross entropy loss function above, 
but we take the softmax of the actuals and weight the predicted output instead.
---------------------------------------------------------------------------------------------------------------
'''
# Weighted (softmax) cross entropy loss
# L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
# or
# L = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(logits=x_vals,
                                                                    labels=targets,
                                                                    pos_weight=weight)

print('xentropy_weighted_y_vals = \n{0}\n'.format(xentropy_weighted_y_vals))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Plot the Categorical Losses                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Plot the output
x_array = x_vals
plt.plot(x_array, hinge_y_vals, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_vals, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_vals, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_vals, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
#plt.xlim(-1, 3)
plt.grid()
plt.legend(loc='lower right', prop={'size': 11})
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '           Softmax entropy and Sparse Entropy                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Since it is hard to graph mutliclass loss functions, we will show how to get the output instead
---------------------------------------------------------------------------------------------------------------
'''
# Softmax entropy loss
# L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits=unscaled_logits,
                        labels=target_dist
                    )
print('softmax_xentropy = {0}\n'.format(softmax_xentropy))

# Sparse entropy loss
# Use when classes and targets have to be mutually exclusive
# L = sum( -actual * log(pred) )
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=unscaled_logits,
                        labels=sparse_target_dist
                    )
print('sparse_xentropy = {0}'.format(sparse_xentropy))

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         loss_function.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()