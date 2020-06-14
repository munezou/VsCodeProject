"""
Using a Multiple Layer Network
------------------------------
We will illustrate how to use a Multiple
Layer Network in TensorFlow

Low Birthrate data:

Columns    Variable                                Abbreviation
----------------------------------------------------------------
Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
                  1 = Birth Weight < 2500g)
Age of the Mother in Years                              AGE
Weight in Pounds at the Last Menstrual Period           LWT
Race (1 = White, 2 = Black, 3 = Other)                  RACE
Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
History of Premature Labor (0 = None  1 = One, etc.)    PTL
History of Hypertension (1 = Yes, 0 = No)               HT
Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
Birth Weight in Grams                                   BWT
-----------------------------------------------------------------

The multiple neural network layer we will create will be composed of
three fully connected hidden layers, with node sizes 50, 25, and 5

"""

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from cpuinfo import get_cpu_info
from packaging import version
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import requests

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

# name of data file
birth_weight_file = 'birth_weight.csv'
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

# Extract y-target (birth weight)
y_vals = np.array([x[8] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest]
                   for x in birth_data])

# set batch size for training
batch_size = 100

# Set random seed to make results reproducible
seed = 4
np.random.seed(seed)
tf.random.set_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices].astype(np.float32)
x_vals_test = x_vals[test_indices].astype(np.float32)
y_vals_train = y_vals[train_indices].astype(np.float32)
y_vals_test = y_vals[test_indices].astype(np.float32)

normal_min =[]
normal_max = []

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

# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random.normal(shape, stddev=st_dev, dtype=tf.float32))
    return weight

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random.normal(shape, stddev=st_dev, dtype= tf.float32))
    return bias

# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    input_layer = tf.cast(input_layer, dtype=tf.float32)
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)

def final_output(
            input_x,
            weight_1, bias_1, 
            weight_2, bias_2, 
            weight_3, bias_3,
            weight_4, bias_4
        ):
    # transrate ndarry to tensor
    input_x = tf.cast(input_x, dtype=tf.float32)

    # -------Create the first layer (50 hidden nodes)--------
    layer_1 = fully_connected(input_x, weight_1, bias_1)

    # -------Create second layer (25 hidden nodes)--------
    layer_2 = fully_connected(layer_1, weight_2, bias_2)

    # -------Create third layer (5 hidden nodes)--------
    layer_3 = fully_connected(layer_2, weight_3, bias_3)

    # -------Create output layer (1 output value)--------
    return fully_connected(layer_3, weight_4, bias_4)


def loss(
            input_x, targets, 
            weight_1, bias_1, 
            weight_2, bias_2, 
            weight_3, bias_3,
            weight_4, bias_4
        ):

    targets = tf.cast(targets, dtype=tf.float32)

    # -------Create output layer (1 output value)--------
    final_output_value = final_output(
                            input_x, 
                            weight_1, bias_1, 
                            weight_2, bias_2, 
                            weight_3, bias_3,
                            weight_4, bias_4
                        )

    # Declare loss function (L1)
    return tf.reduce_mean(tf.abs(targets - final_output_value))

def grad(
            input_x, targets,
            weight_1, bias_1, 
            weight_2, bias_2, 
            weight_3, bias_3,
            weight_4, bias_4
        ):
    with tf.GradientTape() as tape:
        loss_value = loss(
                            input_x, targets,
                            weight_1, bias_1, 
                            weight_2, bias_2, 
                            weight_3, bias_3,
                            weight_4, bias_4
                        )
    return tape.gradient(
                            loss_value, 
                            [
                                weight_1, bias_1, 
                                weight_2, bias_2, 
                                weight_3, bias_3,
                                weight_4, bias_4
                            ]
                        )


# initialize values of 1st Layer
w_1 = init_weight(shape=[7, 25], st_dev=10.0)
b_1 = init_bias(shape=[25], st_dev=10.0)

# initialize values of 2nd Layer
w_2 = init_weight(shape=[25, 10], st_dev=10.0)
b_2 = init_bias(shape=[10], st_dev=10.0)

# initialize values of 3rd Layer
w_3 = init_weight(shape=[10, 3], st_dev=10.0)
b_3 = init_bias(shape=[3], st_dev=10.0)

# initialize values of 4th Layer
w_4 = init_weight(shape=[3, 1], st_dev=10.0)
b_4 = init_bias(shape=[1], st_dev=10.0)

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.8)

# Training loop
loss_vec = []
test_loss = []

for i in range(10000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    grads = grad(rand_x, rand_y, w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4)
    optimizer.apply_gradients(zip(grads, [w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4]))

    temp_loss = loss(rand_x, rand_y, w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4)
    loss_vec.append(temp_loss.numpy())
    
    test_temp_loss = loss(x_vals_test, np.transpose([y_vals_test]), w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4)
    test_loss.append(test_temp_loss.numpy())
    if (i+1) % 250 == 0:
        print('------------------------------------------------------------')
        print('Generation: {0}. Loss = {1}\n'.format(i + 1, temp_loss.numpy()))
        print('weight_1 = \n{0}'.format(w_1.numpy()))
        print('bias_1 = {0}\n'.format(b_1.numpy()))
        print('weight_2 = \n{0}'.format(w_2.numpy()))
        print('bias_2 = {0}\n'.format(b_2.numpy()))
        print('weight_3 = \n{0}'.format(w_3.numpy()))
        print('bias_3 = {0}\n'.format(b_3.numpy()))
        print('weight_4 = \n{0}'.format(w_4.numpy()))
        print('bias_4 = {0}\n\n'.format(b_4.numpy()))


# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# Model Accuracy
actuals = np.array([x[0] for x in birth_data])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]

test_preds = [x[0] for x in final_output(   
                                x_vals_test,
                                w_1, b_1, 
                                w_2, b_2, 
                                w_3, b_3,
                                w_4, b_4
                            )
            ]

train_preds = [x[0] for x in final_output(   
                                x_vals_train,
                                w_1, b_1, 
                                w_2, b_2, 
                                w_3, b_3,
                                w_4, b_4
                            )
            ]

test_preds = np.array([0.0 if x < 2500.0 else 1.0 for x in test_preds])
train_preds = np.array([0.0 if x < 2500.0 else 1.0 for x in train_preds])

# Print out accuracies
test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)])
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))

'''
# Evaluate new points on the model
# Need vectors of 'AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI'
new_data = np.array([[35, 185, 1., 0., 0., 0., 1.],
                     [18, 160, 0., 1., 0., 0., 1.]])

new_data_scaled = np.nan_to_num(normalize_cols(new_data, train_max, train_min))
new_logits = [x[0] for x in sess.run(final_output, feed_dict={x_data: new_data_scaled})]

new_preds = np.array([1.0 if x < 2500.0 else 0.0 for x in new_logits])

print('New Data Predictions: {}'.format(new_preds))

# Evaluations on new/unseen data
'''

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         using_a_multiple_layer_network.py                 ({0})   \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()