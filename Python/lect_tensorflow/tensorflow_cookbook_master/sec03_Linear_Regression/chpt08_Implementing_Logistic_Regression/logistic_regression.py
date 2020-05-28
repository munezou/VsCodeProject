'''
# Logistic Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve logistic regression.
# y = sigmoid(Ax + b)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
import os.path
import csv

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

###
# Obtain and prepare data for modeling
###

# Set name of data file
birth_weight_file = os.path.join(PROJECT_ROOT_DIR, 'birth_weight.csv')

# Download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    with open(birth_weight_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(birth_header)
        writer.writerows(birth_data)
        f.close()

# Read birth weight data into memory
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]

# Pull out target variable
y_vals = np.array([x[0] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
x_vals = np.array([x[1:8] for x in birth_data])

# Set for reproducible results
seed = 99
np.random.seed(seed)
tf.random.set_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Normalize by column (min-max norm)
def normalize_cols(m, col_min=np.array([None]), col_max=np.array([None])):
    if not col_min[0]:
        col_min = m.min(axis=0)
    if not col_max[0]:
        col_max = m.max(axis=0)
    return (m - col_min) / (col_max - col_min), col_min, col_max


x_vals_train, train_min, train_max = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test, _, _ = np.nan_to_num(normalize_cols(x_vals_test, train_min, train_max))

###
# Define Tensorflow computational graph其
###

# Declare batch size
batch_size = 25

# Create variables for linear regression
A = tf.cast(tf.Variable(tf.random.normal(shape=[7,1])), dtype=tf.float32)
b = tf.cast(tf.Variable(tf.random.normal(shape=[1,1])), dtype=tf.float32)

def loss(input_x, aa, bb, targets):
    # Declare model operations
    model_output = tf.add(tf.matmul(input_x, aa), bb)

    # Declare loss function (Cross Entropy loss)
    return tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=targets))

def grad(input_x, aa, bb, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, aa, bb, targets)
    return tape.gradient(loss_value, [aa, bb])

def accuracy(input_x, aa, bb, targets):
    # Declare model operations
    model_output = tf.add(tf.matmul(input_x, aa), bb)

    prediction = tf.round(tf.sigmoid(model_output))
    predictions_correct = tf.cast(tf.equal(prediction, targets), tf.float32)
    return tf.reduce_mean(input_tensor=predictions_correct) 

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

###
# Train model
###

# Training loop
loss_vec = []
train_acc = []
test_acc = []

for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x_train = tf.cast(x_vals_train[rand_index], dtype=tf.float32)
    rand_y_train = tf.cast(np.transpose([y_vals_train[rand_index]]), dtype=tf.float32)

    grads = grad(rand_x_train, A, b, rand_y_train)
    optimizer.apply_gradients(zip(grads, [A, b]))

    temp_loss = loss(rand_x_train, A, b, rand_y_train)
    loss_vec.append(temp_loss)
    
    # Actual Prediction(train)
    temp_acc_train = accuracy(tf.cast(x_vals_train, dtype=tf.float32), A, b, tf.cast(np.transpose([y_vals_train]), dtype=tf.float32))
    train_acc.append(temp_acc_train)

    # Actual Prediction(test)
    temp_acc_test = accuracy(tf.cast(x_vals_test, dtype=tf.float32), A, b, tf.cast(np.transpose([y_vals_test]), dtype=tf.float32))
    test_acc.append(temp_acc_test)

    if (i + 1) % 300 == 0:
        print('Step #{0}\n  A = {1},  b = {2}'.format(i + 1, A.numpy(), b.numpy()))
        print('Loss = {0}\n'.format(temp_loss.numpy()))

###
# Display model performance
###

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

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
        '       finished         logistic_regression.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()