'''
# SVM Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve support vector regression. We are going
# to find the line that has the maximum margin
# which INCLUDES as many points as possible
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Width
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


# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# Split data into train/test sets
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 50

epsilon = tf.cast(tf.constant([0.5]), dtype=tf.float32)

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)

def loss(input_x, aa, bb, eps, targets):
    # Declare model operations
    model_output = tf.add(tf.matmul(input_x, aa), bb)

    # Declare loss function
    # = max(0, abs(target - predicted) + epsilon)
    # 1/2 margin width parameter = epsilon
    epsilon = tf.constant([0.5])

    # Margin term in loss
    return tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, targets)), eps)))

def grad(input_x, aa, bb, eps, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, aa, bb, eps, targets)
    return tape.gradient(loss_value, [aa, bb])


# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.052)

# Training loop
train_loss = []
test_loss = []

for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = tf.cast(np.transpose([x_vals_train[rand_index]]), dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals_train[rand_index]]), dtype=tf.float32)


    grads = grad(rand_x, A, b, epsilon, rand_y)
    optimizer.apply_gradients(zip(grads, [A, b]))
    
    temp_train_loss = loss(tf.cast(np.transpose([x_vals_train]), dtype=tf.float32), A, b, epsilon, tf.cast(np.transpose(y_vals_train), dtype=tf.float32))
    train_loss.append(temp_train_loss)
    
    temp_test_loss = loss(tf.cast(np.transpose([x_vals_test]), dtype=tf.float32), A, b, epsilon, tf.cast(np.transpose(y_vals_test), dtype=tf.float32))
    test_loss.append(temp_test_loss)
    if (i + 1) % 50 == 0:
        print('-----------')
        print('Generation: {0}'.format(i + 1))
        print('A = {0},  b = {1}'.format(A.numpy(), b.numpy()))
        print('Train Loss = {0}'.format(temp_train_loss.numpy()))
        print('Test Loss = {0}'.format(temp_test_loss.numpy()))

# Extract Coefficients
[[slope]] = A.numpy()
[[y_intercept]] = b.numpy()
width = epsilon.numpy()

# Get best fit line
best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
    best_fit_upper.append(slope*i+y_intercept+width)
    best_fit_lower.append(slope*i+y_intercept-width)

# Plot fit with data
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(train_loss, 'k-', label='Train Set Loss')
plt.plot(test_loss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         support_vector_regression.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()