'''
# Linear Regression: L1 vs L2
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression via the matrix inverse.
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from packaging import version
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

print(__doc__)

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare batch size and number of iterations
batch_size = 25
learning_rate = 0.4 # Will not converge with learning rate at 0.4
iterations = 50

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)

# L1 loss
def loss_l1(input_x1, aa1, bb1, targets1):
    # Declare model operations
    model_output_l1 = tf.add(tf.matmul(input_x1, aa1), bb1)

    # Declare loss functions
    return tf.reduce_mean(tf.abs(targets1 - model_output_l1))

def grad_l1(input_x1, aa1, bb1, targets1):
    with tf.GradientTape() as tape1:
        loss_value_l1 = loss_l1(input_x1, aa1, bb1, targets1)
    return tape1.gradient(loss_value_l1, [aa1, bb1])

# Declare optimizers
opt_l1 = tf.keras.optimizers.SGD(learning_rate)

# Training loop
loss_vec_l1 = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = tf.cast(np.transpose([x_vals[rand_index]]), dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
    grads_l1 = grad_l1(rand_x, A, b, rand_y)

    opt_l1.apply_gradients(zip(grads_l1, [A, b]))

    temp_loss_l1 = loss_l1(rand_x, A, b, rand_y)
    loss_vec_l1.append(temp_loss_l1)
    if (i+1)%25==0:
        print('Step #{0}   A = {1},  b = {2}'.format(i+1, A.numpy(), b.numpy()))


# L2 Loss
def loss_l2(input_x2, aa2, bb2, targets2):
    # Declare model operations
    model_output_l2 = tf.add(tf.matmul(input_x2, aa2), bb2)

    # Declare loss functions
    return tf.reduce_mean(tf.square(targets2 - model_output_l2))

def grad_l2(input_x2, aa2, bb2, targets2):
    with tf.GradientTape() as tape2:
        loss_value_l2 = loss_l2(input_x2, aa2, bb2, targets2)
    return tape2.gradient(loss_value_l2, [aa2, bb2])

# Declare optimizers
opt_l2 = tf.keras.optimizers.SGD(learning_rate)

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)


# Training loop
loss_vec_l2 = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = tf.cast(np.transpose([x_vals[rand_index]]), dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
    grads_l2 = grad_l2(rand_x, A, b, rand_y)

    opt_l2.apply_gradients(zip(grads_l2, [A, b]))

    temp_loss_l2 = loss_l2(rand_x, A, b, rand_y)
    loss_vec_l2.append(temp_loss_l2)
    if (i+1)%25==0:
        print('Step #{0}  A = {1},  b = {2}'.format(i+1, A.numpy(), b.numpy()))


# Plot loss over time
plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
plt.title('L1 and L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L1 Loss')
plt.legend(loc='upper right')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         lin_reg_l1_vs_l2.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()