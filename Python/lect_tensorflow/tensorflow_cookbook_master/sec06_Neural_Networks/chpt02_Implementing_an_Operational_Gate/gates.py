'''
# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in TensorFlow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask TensorFlow to change the
# variable based on our loss function
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import tensorflow as tf
from tensorflow.python.framework import ops

print(__doc__)

print(
    '--------------------------------------------------------------------------\n'
    '                      cpu information                                     \n'
    '--------------------------------------------------------------------------\n'
)
# display the using cpu information
for key, value in get_cpu_info().items():
    print("{0}: {1}".format(key, value))

print()
print()

# Display current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: {0}\n".format(tf.version.VERSION))
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

#----------------------------------
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#

a = tf.cast(tf.Variable(tf.constant(4.)), dtype=tf.float32)
x_val = tf.cast(5., dtype=tf.float32)
target = tf.cast(50., dtype=tf.float32)

def multiplication(aa, input_x):
    return tf.multiply(aa, input_x)

def loss(aa, input_x, targets):
    # Declare the loss function as the difference between
    # the output and a target value, 50.
    multiple_value = multiplication(aa, input_x)
    return tf.square(tf.subtract(multiple_value, targets))

def grad(aa, input_x, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(aa, input_x, targets)
    return tape.gradient(loss_value, [aa])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(0.01)

# Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')

for _ in range(10):
    grads = grad(a, x_val, target)
    optimizer.apply_gradients(zip(grads, [a]))

    a_val = a

    mult_output = multiplication(a_val, x_val)

    print(str(a_val.numpy()) + ' * ' + str(x_val.numpy()) + ' = ' + str(mult_output.numpy()))


'''
Create a nested gate:
   f(x) = a * x + b

    a --
        |
        |-- (multiply)--
    x --|               |
                        |-- (add) --> output
                    b --|

'''

a = tf.cast(tf.Variable(tf.constant(1.)), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.constant(1.)), dtype=tf.float32)
x_val = tf.cast(5., dtype=tf.float32)
target = tf.cast(50., dtype=tf.float32)

def two_gate(input_x, aa, bb):
    return tf.add(tf.multiply(aa, input_x), bb)

def loss_two(input_x, targets, aa, bb):
    two_gate_value = two_gate(input_x, aa, bb)

    # Declare the loss function as the difference between
    # the output and a target value, 50.
    return tf.square(tf.subtract(two_gate_value, targets))

def grad_two(input_x, targets, aa, bb):
    with tf.GradientTape() as tape:
        loss_value = loss_two(input_x, targets, aa, bb)
    return tape.gradient(loss_value, [aa, bb])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(0.01)

# Run loop across gate
print('\nOptimizing Two Gate Output to 50.')

for _ in range(10):
    grads = grad_two(x_val, target, a, b)
    optimizer.apply_gradients(zip(grads, [a, b]))

    a_val = a 
    b_val = b

    two_gate_output = two_gate(x_val, a_val, b_val)
    
    print(str(a_val.numpy()) + ' * ' + str(x_val.numpy()) + ' + ' + str(b_val.numpy()) + ' = ' + str(two_gate_output.numpy()))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      gates.py                          ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()