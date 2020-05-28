'''
# Linear Regression: TensorFlow Way
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width
'''

# common library
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

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Declare batch size
batch_size = 25

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)

def loss(x_input, aa, bb, targets):
    # Declare model operations
    model_output = tf.add(tf.matmul(x_input, aa), bb)

    # Declare loss function (L2 loss)
    return tf.reduce_mean(tf.square(targets - model_output))

def grad(x_input, aa, bb, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(x_input, aa, bb, targets)
    return tape.gradient(loss_value, [aa, bb])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

# Training loop
loss_vec = []

for i in range(100):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = tf.cast(np.transpose([x_vals[rand_index]]), dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
    grads = grad(rand_x, A, b, rand_y)

    optimizer.apply_gradients(zip(grads, [A, b]))
    temp_loss = loss(rand_x, A, b, rand_y)
    loss_vec.append(temp_loss)
    if (i+1)%25==0:
        print('Step #{0}  A = {1},  b = {2}'.format(i+1, A.numpy(), b.numpy()))
        print('Loss = {0}\n'.format(temp_loss))

# Get the optimal coefficients
[slope] = A.numpy()
[y_intercept] = b.numpy()

# Get best fit line
best_fit = []

for i in x_vals:
    best_fit.append(slope*i+y_intercept)

# Plot the result
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         lin_reg_tensorflow_way.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()