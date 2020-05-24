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
print('tf.math.floordiv = {0}\n'.format(tf.math.floordiv(3.0, 4.0)))

# Mod function
print('tf.math.mod(22.0, 5.0) = {0}'.format(tf.math.mod(22.0, 5.0)))

# Cross product
print('tf.linalg.cross([1.,0.,0.],[0.,1.,0.]) = {0}'.format(tf.linalg.cross([1.,0.,0.],[0.,1.,0.])))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Trig functions                                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# sine, cosine, and tangent
print('tf.math.sin(3.1416) = {0}'.format(tf.math.sin(3.1416)))
print('tf.math.cos(3.1416) = {0}'.format(tf.math.cos(3.1416)))
print('tf.math.tan(3.1416/4.) = {0}'.format(tf.math.tan(3.1416/4.)))

# trig functions
# custom operation
test_nums = range(15)

def custom_polynomial(x_val):
    # return 3x^2 - a + 10
    return tf.subtract(3 * tf.square(x_val), x_val) + 10

print('custom_polynomial(11) = {0}'.format(custom_polynomial(11)))

# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print(expected_output)

# TensorFlow custom function output
for num in test_nums:
    print('coustom_polynomial({0}) = {1}'.format(num, custom_polynomial(num)))

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         operations.py                                  ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()