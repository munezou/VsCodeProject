'''
# Mixed Distance Functions for  k-Nearest Neighbor
#----------------------------------
#
# This function shows how to use different distance
# metrics on different features for kNN.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's
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

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

## Create distance metric weight matrix weighted by standard deviation
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.linalg.tensor_diag(weight_diagonal), dtype=tf.float32)

# Split the data into train and test sets
np.random.seed(13)   # reproducible results
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size=len(x_vals_test)

def distance(input_x_train, input_x_test, w_matrix, bat_size):
    # Declare weighted distance metric
    # Weighted L2 = sqrt((x-y)^T * A * (x-y))
    subtraction_term =  tf.subtract(input_x_train, tf.expand_dims(input_x_test, 1))
    first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(w_matrix, 0), [bat_size, 1, 1]))
    second_product = tf.matmul(first_product, tf.transpose(subtraction_term, perm=[0, 2, 1]))
    return tf.sqrt(tf.linalg.diag_part(second_product))

def prediction(input_x_train, input_x_test, targets_train, w_matrix, bat_size, kk):
    tmp_distance = distance(input_x_train, input_x_test, w_matrix, bat_size)

    # Predict: Get min distance index (Nearest neighbor)
    top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(tmp_distance), k=kk)
    x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
    x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))
    x_val_weights = tf.expand_dims(tf.math.divide(top_k_xvals, x_sums_repeated), 1)

    top_k_yvals = tf.gather(targets_train, top_k_indices)
    return tf.squeeze(tf.matmul(x_val_weights, top_k_yvals), axis=[1])

def mse(input_x_train, input_x_test, targets_train, targets_test, w_matrix, bat_size, kk):
    # Calculate MSE
    temp_prediction = prediction(input_x_train, input_x_test, targets_train, w_matrix, bat_size, kk)
    return tf.math.divide(tf.reduce_sum(tf.square(tf.subtract(temp_prediction, targets_test))), bat_size)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size,len(x_vals_train))

    x_batch = tf.cast(x_vals_test[min_index:max_index], dtype=tf.float32)
    y_batch = tf.cast(y_vals_test[min_index:max_index], dtype=tf.float32)

    predictions = prediction(tf.cast(x_vals_train, dtype=tf.float32), x_batch, tf.cast(y_vals_train, dtype=tf.float32), weight_matrix, batch_size, k)
    
    batch_mse = mse(tf.cast(x_vals_train, dtype=tf.float32), x_batch, tf.cast(y_vals_train, dtype=tf.float32), y_batch, weight_matrix, batch_size, k)

    print('Batch #{0}  MSE: {1}\n'.format(i + 1, (np.round(batch_mse.numpy(),3))))

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)

plt.hist(predictions.numpy(), bins, alpha=0.5, label='Prediction')
plt.hist(y_batch.numpy(), bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished        mixed_distance_functions_knn.py                           ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()