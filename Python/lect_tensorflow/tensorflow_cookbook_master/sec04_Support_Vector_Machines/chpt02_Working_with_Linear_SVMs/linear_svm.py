# Linear Support Vector Machine: Soft Margin
# ----------------------------------
#
# This function shows how to use TensorFlow to
# create a soft margin SVM
#
# We will use the iris data, specifically:
#  x1 = Sepal Length
#  x2 = Petal Width
# Class 1 : I. setosa
# Class -1: not I. setosa
#
# We know here that x and y are linearly seperable
# for I. setosa classification.

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Set random seeds
np.random.seed(7)
tf.random.set_seed(7)

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

# Split data into train/test sets
train_indices = np.random.choice(
                    len(x_vals),
                    int(round(len(x_vals)*0.9)),
                    replace=False
                )

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 135

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[2, 1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1, 1])), dtype=tf.float32)

def loss(input_x, aa, bb, targets):
    # Declare model operations
    model_output_loss = tf.subtract(tf.matmul(input_x, aa), bb)

    # Declare vector L2 'norm' function squared
    l2_norm = tf.reduce_sum(tf.square(aa))

    # Declare loss function
    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    # L2 regularization parameter, alpha
    alpha = tf.constant([0.01])

    # Margin term in loss
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output_loss, targets))))

    # Put terms together
    return tf.add(classification_term, tf.multiply(alpha, l2_norm))

def grad(input_x, aa, bb, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, aa, bb, targets)
    return tape.gradient(loss_value, [aa, bb])

def accuracy(input_x, aa, bb, targets):
    # Declare model operations
    model_output_acc = tf.subtract(tf.matmul(input_x, aa), bb)

    # Declare prediction function
    prediction = tf.sign(model_output_acc)
    return tf.reduce_mean(tf.cast(tf.equal(prediction, targets), tf.float32))  

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = tf.cast(x_vals_train[rand_index], dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals_train[rand_index]]), dtype=tf.float32)

    grads = grad(rand_x, A, b, rand_y)
    optimizer.apply_gradients(zip(grads, [A, b]))

    temp_loss = loss(rand_x, A, b, rand_y)
    loss_vec.append(temp_loss)

    train_acc_temp = accuracy(tf.cast(x_vals_train, dtype=tf.float32), A, b, tf.cast(np.transpose([y_vals_train]), dtype=tf.float32))
    train_accuracy.append(train_acc_temp)

    test_acc_temp = accuracy(tf.cast(x_vals_test, dtype=tf.float32), A, b, tf.cast(np.transpose([y_vals_test]), dtype=tf.float32))
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #{0} A = {1}, b = {2}'.format(i + 1, A.numpy(), b.numpy()))
        print('Loss = {0}\n'.format(temp_loss.numpy()))

# Extract coefficients
[[a1], [a2]] = A.numpy()
[[b]] = b.numpy()
slope = -a2/a1
y_intercept = b/a1

# Extract x1 and x2 vals
x1_vals = [d[1] for d in x_vals]

# Get best fit line
best_fit = []
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)

# Separate I. setosa
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

# Plot data and line
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot train/test accuracies
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

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
        '       finished         linear_svm.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()