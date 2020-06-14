'''
# MNIST Digit Prediction with k-Nearest Neighbors
#-----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x8 matrices
# of floating point numbers
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

print(__doc__)

'''
--------------------------------------------
In casee of windows, os name is 'nt'.
In case of linux, os name is 'posix'.
--------------------------------------------
'''

if os.name == 'nt':
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

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# transrate y_train by one-hot encode.
n_train_label = len(np.unique(y_train))
y_train = np.eye(n_train_label, dtype=np.uint8)[y_train]

n_test_lavel = len(np.unique(y_test))
y_test = np.eye(n_test_lavel, dtype=np.uint8)[y_test]


# Random sample
np.random.seed(13)  # set seed for reproducibility
train_size = 1000
test_size = 102

rand_train_indices = np.random.choice(len(x_train), train_size, replace=False)
rand_test_indices = np.random.choice(len(x_test), test_size, replace=False)

x_vals_train = x_train[rand_train_indices].reshape([train_size, 784])
x_vals_test = x_test[rand_test_indices].reshape([test_size, 784])

y_vals_train = y_train[rand_train_indices]
y_vals_test = y_test[rand_test_indices]

# Declare k-value and batch size
k = 4
batch_size = 6
kind_loss = 'L1'
#kind_loss = 'L2'

def prediction(loss, input_x_train, targets_train, input_x_test, targets_test, kk):
    # Declare distance metric
    if loss == 'L1':
        # L1
        distance = tf.reduce_sum(input_tensor=tf.abs(tf.subtract(input_x_train, tf.expand_dims(input_x_test, 1))), axis=2)
    elif loss == 'L2':
        # L2
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(input_x_train, tf.expand_dims(input_x_test, 1))), reduction_indices=1))
    else:
        print('error')

    # Predict: Get min distance index (Nearest neighbor)
    top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=kk)
    prediction_indices = tf.gather(targets_train, top_k_indices)

    # Predict the mode category
    count_of_predictions = tf.reduce_sum(input_tensor=prediction_indices, axis=1)
    return tf.argmax(input=count_of_predictions, axis=1)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test) / batch_size))

test_output = []
actual_vals = []

x_batchs = []
y_batchs = []

for i in range(num_loops):
    min_index = i * batch_size
    max_index = min((i + 1) * batch_size, len(x_vals_train))

    x_batch = tf.cast(x_vals_test[min_index:max_index], dtype=tf.float32)
    y_batch = tf.cast(y_vals_test[min_index:max_index], dtype=tf.float32)

    predictions = prediction(kind_loss, tf.cast(x_vals_train, dtype=tf.float32), tf.cast(y_vals_train, dtype=tf.float32), x_batch, x_batch, k)

    test_output.extend(predictions)
    actual_vals.extend(np.argmax(y_batch, axis=1))

    x_batchs.extend(x_batch)
    y_batchs.extend(y_batch)

accuracy = sum([1. / test_size for i in range(test_size) if test_output[i] == actual_vals[i]])
print('Accuracy on test set: {0}\n'.format(accuracy))

# Plot the last batch results:

Nrows = 2
Ncols = 3

actual = Nrows * Ncols

plt.figure(figsize=(12, 8))

for i in range(actual):
    plt.subplot(Nrows, Ncols, i + 1)
    plt.imshow(np.reshape(x_batchs[i], [28, 28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actual_vals[i]) + ' Pred: ' + str(test_output[i].numpy()), fontsize=6)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      image_recognition.py                          ({0})          \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()