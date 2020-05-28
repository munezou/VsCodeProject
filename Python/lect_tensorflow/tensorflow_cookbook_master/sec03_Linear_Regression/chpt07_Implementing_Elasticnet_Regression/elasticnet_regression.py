'''
# Elastic Net Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve elastic net regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

###
# Obtain data
###

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

###
# Setup model
###

# make results reproducible
seed = 13
np.random.seed(seed)
tf.random.set_seed(seed)

# Declare batch size
batch_size = 50

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[3,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)

def loss(input_x, aa, bb, targets):
    # Declare model operations
    model_output = tf.add(tf.matmul(input_x, aa), bb)

    # Declare the elastic net loss function
    elastic_param1 = tf.constant(1.)
    elastic_param2 = tf.constant(1.)
    l1_a_loss = tf.reduce_mean(tf.abs(aa))
    l2_a_loss = tf.reduce_mean(tf.square(aa))
    e1_term = tf.multiply(elastic_param1, l1_a_loss)
    e2_term = tf.multiply(elastic_param2, l2_a_loss)

    return tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(targets - model_output)), e1_term), e2_term), 0)

def grad(input_x, aa, bb, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, aa, bb, targets)
    return tape.gradient(loss_value, [aa, bb])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(0.001)

###
# Train model
###

# Training loop
loss_vec = []

for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = tf.cast(x_vals[rand_index], dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
    grads = grad(rand_x, A, b, rand_y)

    optimizer.apply_gradients(zip(grads, [A, b]))

    temp_loss = loss(rand_x, A, b, rand_y)
    loss_vec.append(temp_loss[0])
    if (i + 1) % 250 == 0:
        print('Step #{0}\n  A = {1},  b = {2}'.format(i + 1, A.numpy(), b.numpy()))
        print('Loss = {0}\n'.format(temp_loss.numpy()))

###
# Extract model results
###

# Get the optimal coefficients
[[sw_coef], [pl_coef], [pw_ceof]] = A.numpy()
[y_intercept] = b.numpy()

###
# Plot results
###

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         elasticnet_regression.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()