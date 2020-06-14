"""
Improving Linear Regression with Neural Networks (Logistic Regression)
----------------------------------------------------------------------

This function shows how to use TensorFlow to
solve logistic regression with a multiple layer neural network
y = sigmoid(A3 * sigmoid(A2* sigmoid(A1*x + b1) + b2) + b3)

We will use the low birth weight data, specifically:
 y = 0 or 1 = low birth weight
 x = demographic and medical history data
"""

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
import os.path
import csv

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

# Name of data file
birth_weight_file = os.path.join(PROJECT_ROOT_DIR, 'birth_weight.csv')
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master' \
                '/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'

# Download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1]
                  for y in birth_data[1:] if len(y) >= 1]
    with open(birth_weight_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)

# read birth weight data into memory
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        if len(row) >= 9:
            birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

# Pull out target variable
y_vals = np.array([x[0] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([x[1:8] for x in birth_data])

# Set random seed for reproducible results
seed = 99
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Declare batch size
batch_size = 90

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

normal_min = []
normal_max = []

# Normalize by column (min-max norm to be between 0 and 1)
# min-max normalization
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    normal_min.append(min)
    max = x.max(axis=axis, keepdims=True)
    normal_max.append(max)
    result = (x-min)/(max-min)
    return result

def min_max_2(x, min, max, axi=None):
    result = (x - min) / (max - min)


# Calculate median value in column
x_vals_train_medien = np.median(x_vals_train, axis=0)
print('x_vals_train_medien = \n{0}'.format(x_vals_train_medien))

x_vals_test_medien = np.median(x_vals_test, axis=0)
print('x_vals_train_medien = \n{0}'.format(x_vals_test_medien))

# When the Nan value is detected, enter the median value.
for tmpRow in x_vals_train:
    if np.isnan(tmpRow.any()) == True:
        for j in range(len(tmpRow)):
            tmpRow[j] = x_vals_train_medien[j]

for tmpRow in x_vals_test:
    if np.isnan(tmpRow.any()) == True:
        for j in range(len(tmpRow)):
            tmpRow[j] = x_vals_test_medien[j]

# Perform standardization.
x_vals_train = min_max(x_vals_train, axis=0)
x_vals_test = min_max(x_vals_test, axis=0)

# Create variable definition
def init_variable(shape):
    return tf.Variable(tf.random.normal(shape=shape, dtype=tf.float32))


# Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation=True):
    input_layer = tf.cast(input_layer, dtype=tf.float32)

    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    # We separate the activation at the end because the loss function will
    # implement the last sigmoid necessary
    if activation:
        return tf.nn.sigmoid(linear_layer)
    else:
        return linear_layer

def final_output(input_data, aa1, bb1, aa2, bb2, aa3, bb3):
    # First logistic layer (7 inputs to 7 hidden nodes)
    logistic_layer1 = logistic(input_data, aa1, bb1)

    # Second logistic layer (7 hidden inputs to 5 hidden nodes)
    logistic_layer2 = logistic(logistic_layer1, aa2, bb2)

    # Final output layer (5 hidden nodes to 1 output)
    return logistic(logistic_layer2, aa3, bb3, activation=False)

def loss(input_data, targets, aa1, bb1, aa2, bb2, aa3, bb3):
    # Match the variable types.
    input_data = tf.cast(input_data, dtype= tf.float32)
    targets = tf.cast(targets, dtype=tf.float32)

    # calculate final_output
    final_output_value = final_output(input_data, aa1, bb1, aa2, bb2, aa3, bb3)

    # Declare loss function (Cross Entropy loss)
    return tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output_value, labels=targets))

def grad(input_x, targets, aa1, bb1, aa2, bb2, aa3, bb3):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, targets, aa1, bb1, aa2, bb2, aa3, bb3)
    return tape.gradient(loss_value, [aa1, bb1, aa2, bb2, aa3, bb3])

# Initialize variable in 1st Layer
A1 = init_variable(shape=[7, 14])
b1 = init_variable(shape=[14])

# Initialize variable in 2nd Layer
A2 = init_variable(shape=[14, 5])
b2 = init_variable(shape=[5])

# Initialize variable in 3rd Layer
A3 = init_variable(shape=[5, 1])
b3 = init_variable(shape=[1])

# Declare optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

def accuracy(input_data, targets, aa1, bb1, aa2, bb2, aa3, bb3):
    # Actual Prediction
    prediction = tf.round(tf.nn.sigmoid(final_output(input_data, aa1, bb1, aa2, bb2, aa3, bb3)))
    predictions_correct = tf.cast(tf.equal(prediction, targets), tf.float32)
    return tf.reduce_mean(input_tensor=predictions_correct)

# Training loop
loss_vec = []
train_acc = []
test_acc = []

for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    grads = grad(rand_x, rand_y, A1, b1, A2, b2, A3, b3)
    optimizer.apply_gradients(zip(grads, [A1, b1, A2, b2, A3, b3]))

    temp_loss = loss(rand_x, rand_y, A1, b1, A2, b2, A3, b3)
    loss_vec.append(temp_loss.numpy())

    temp_acc_train = accuracy(x_vals_train, np.transpose([y_vals_train]), A1, b1, A2, b2, A3, b3)
    train_acc.append(temp_acc_train.numpy())

    temp_acc_test = accuracy(x_vals_test, np.transpose([y_vals_test]), A1, b1, A2, b2, A3, b3)
    test_acc.append(temp_acc_test.numpy())

    if (i + 1) % 150 == 0:
        print('Loss = {}'.format(temp_loss.numpy()))

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

plt.close()

# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         improving_linear_regression.py                 ({0})   \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()