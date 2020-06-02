'''
Combining Gates and Activation Functions

This function shows how to implement
various gates with activation functions
in TensorFlow

This function is an extension of the
prior gates, but with various activation
functions.
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

# specificate random seed.
tf.random.set_seed(5)
np.random.seed(42)

# specificate batch size
batch_size = 50

# Argument setting
a1 = tf.cast(tf.Variable(tf.random.normal(shape=[1, 1])), dtype=tf.float32)
b1 = tf.cast(tf.Variable(tf.random.uniform(shape=[1, 1])), dtype=tf.float32)
a2 = tf.cast(tf.Variable(tf.random.normal(shape=[1, 1])), dtype=tf.float32)
b2 = tf.cast(tf.Variable(tf.random.uniform(shape=[1, 1])), dtype=tf.float32)

# raw data
x = np.random.normal(2, 0.1, 500)

target = tf.cast(0.75, dtype=tf.float32)

# Declare the loss function as the difference between
# the output and a target value, 0.75.
def sigmoid_activation(input_x, aa1, bb1):
    return tf.sigmoid(tf.add(tf.matmul(input_x, aa1), bb1))

def loss1(input_x, targets, aa1, bb1):
    sigmoid_activation_value = sigmoid_activation(input_x, aa1, bb1)
    return tf.reduce_mean(input_tensor=tf.square(tf.subtract(sigmoid_activation_value, targets)))

def grad1(input_x, targets, aa1, bb1):
    with tf.GradientTape() as tape:
        loss_value = loss1(input_x, targets, aa1, bb1)
    return tape.gradient(loss_value, [aa1, bb1])

def relu_activation(input_x, aa2, bb2):
    return tf.nn.relu(tf.add(tf.matmul(input_x, aa2), bb2))

def loss2(input_x, targets, aa2, bb2):
    relu_activation_value = relu_activation(input_x, aa2, bb2)
    return tf.reduce_mean(input_tensor=tf.square(tf.subtract(relu_activation_value, targets)))

def grad2(input_x, targets, aa2, bb2):
    with tf.GradientTape() as tape:
        loss_value = loss2(input_x, targets, aa2, bb2)
    return tape.gradient(loss_value, [aa2, bb2])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(0.01)

# Run loop across gate
print('\nOptimizing Sigmoid AND Relu Output to 0.75')

loss_vec_sigmoid = []
loss_vec_relu = []

activation_sigmoid = []
activation_relu = []

for i in range(500):
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = tf.cast(np.transpose([x[rand_indices]]), dtype=tf.float32)

    grads1 = grad1(x_vals, target, a1, b1)
    optimizer.apply_gradients(zip(grads1, [a1, b1]))

    grads2 = grad2(x_vals, target, a2, b2)
    optimizer.apply_gradients(zip(grads2, [a2, b2]))
    
    loss_vec_sigmoid.append(loss1(x_vals, target, a1, b1).numpy())
    loss_vec_relu.append(loss2(x_vals, target, a2, b2).numpy())    
    
    sigmoid_output = np.mean(sigmoid_activation(x_vals, a1, b1).numpy())
    relu_output = np.mean(relu_activation(x_vals, a2, b2).numpy())

    activation_sigmoid.append(sigmoid_output)
    activation_relu.append(relu_output)
    
    if i % 50 == 0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))

# plot Activation function
plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(activation_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()

# Plot the loss
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      activation_functions.py                ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()