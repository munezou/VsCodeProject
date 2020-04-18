'''
**************************************************
Operations
----------------------------------

    This function introduces various operations
    in TensorFlow
**************************************************
'''
# Declaring Operations
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import datetime
from packaging import version
import tensorflow as tf

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       function                                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# div() vs truediv() vs floordiv()
print('tf.math.divide(3, 4) = {0}'.format(tf.math.divide(3, 4)))
print('tf.math.truediv = {0}'.format(tf.math.truediv(3, 4)))
print('tf.math.floordiv = {0}\n'.format(tf.math.floordiv((3, 4))))

# Mod function

'''
# Mod function
print(sess.run(tf.mod(22.0, 5.0)))

# Cross Product
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))

# Trig functions
print(sess.run(tf.sin(3.1416)))
print(sess.run(tf.cos(3.1416)))
print(sess.run(tf.tan(3.1416/4.)))

# Custom operation
test_nums = range(15)


def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return tf.subtract(3 * tf.square(x_val), x_val) + 10

print(sess.run(custom_polynomial(11)))

# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print(expected_output)

# TensorFlow custom function output
for num in test_nums:
    print(sess.run(custom_polynomial(num)))
'''
