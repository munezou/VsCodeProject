'''
Implementing a one-layer Neural Network

We will illustrate how to create a one hidden layer NN

We will use the iris data for this exercise

We will build a one-hidden layer neural network
 to predict the fourth attribute, Petal Width from
 the other three (Sepal length, Sepal width, Petal length).
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

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

# Load iris data
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

# make results reproducible
seed = 2
tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]

y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))

x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare batch size
batch_size = 50

# Create variables for both NN layers
hidden_layer_nodes = 10

A1 = tf.cast(tf.Variable(tf.random.normal(shape=[3, hidden_layer_nodes])), dtype=tf.float32)  # inputs -> hidden nodes
b1 = tf.cast(tf.Variable(tf.random.normal(shape=[hidden_layer_nodes])), dtype=tf.float32)  # one biases for each hidden node
A2 = tf.cast(tf.Variable(tf.random.normal(shape=[hidden_layer_nodes, 1])), dtype=tf.float32)  # hidden inputs -> 1 output
b2 = tf.cast(tf.Variable(tf.random.normal(shape=[1])), dtype=tf.float32)  # 1 bias for the output

def loss(input_x, targets, aa1, bb1, aa2, bb2):
    # Declare model operations
    hidden_output = tf.nn.relu(tf.add(tf.matmul(input_x, aa1), bb1))
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, aa2), bb2))

    # Declare loss function (MSE)
    return tf.reduce_mean(input_tensor=tf.square(targets - final_output))

def grad(input_x, targets, aa1, bb1, aa2, bb2):
    with tf.GradientTape() as tape:
        loss_value = loss(input_x, targets, aa1, bb1, aa2, bb2)
    return tape.gradient(loss_value, [aa1, bb1, aa2, bb2])

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(0.005)

# Training loop
loss_vec = []
test_loss = []

for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = tf.cast(x_vals_train[rand_index], dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals_train[rand_index]]), dtype=tf.float32)

    grads = grad(rand_x, rand_y, A1, b1, A2, b2)
    optimizer.apply_gradients(zip(grads, [A1, b1, A2, b2]))

    temp_loss = loss(rand_x, rand_y, A1, b1, A2, b2)
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = loss(tf.cast(x_vals_test, dtype=tf.float32), tf.cast(np.transpose([y_vals_test]), dtype=tf.float32), A1, b1, A2, b2)
    test_loss.append(np.sqrt(test_temp_loss))
    if (i + 1) % 50 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss.numpy()))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      single_hidden_layer_network.py                ({0})             \n'.format(date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()