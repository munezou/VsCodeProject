'''
# Evaluating models in TensorFlow
#
# This code will implement two models.  The first
#  is a simple regression model, we will show how to
#  call the loss function, MSE during training, and
#  output it after for test and training sets.
#
# The second model will be a simple classification
#  model.  We will also show how to print percent
#  classified correctly during training and after
#  for both the test and training sets.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(__doc__)

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."


'''
# Regression Example:
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.
'''

# Declare batch size
batch_size = 25

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

def loss(x_data, a, targets):
    my_out = tf.matmul(x_data, a)
    return tf.reduce_mean(tf.square(my_out - targets))

def grad(x_data, a, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(x_data, a, targets)
    return tape.gradient(loss_value, [a])

# Create variable (one model parameter = A)
A = tf.Variable(tf.random.normal(shape=[1,1]))

# decide optimize method.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Run Loop
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = tf.cast(np.transpose([x_vals_train[rand_index]]), dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals_train[rand_index]]), dtype=tf.float32)
    
    grands = grad(rand_x, A, rand_y)

    optimizer.apply_gradients(zip(grands, [A]))

    if (i+1)%25==0:
        print('Step # {0} A = {1}'.format(i + 1, A.numpy()))
        tmploss = loss(rand_x, A, rand_y)
        print('Loss = {0}'.format(tmploss))


# Evaluate accuracy (loss) on test set
mse_test = loss(tf.cast(np.transpose([x_vals_test]), dtype=tf.float32), A, tf.cast(np.transpose([y_vals_test]), dtype=tf.float32))
mse_train = loss(tf.cast(np.transpose([x_vals_train]), dtype=tf.float32), A, tf.cast(np.transpose([y_vals_train]), dtype=tf.float32))
print('MSE on test:' + str(np.round(mse_test, 2)))
print('MSE on train:' + str(np.round(mse_train, 2)))

# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

# Declare batch size
batch_size = 25

# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

def loss2(x_data2, a2, targets2):
    my_out2 = tf.add(x_data2, a2)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_out2, labels=targets2))

def grad2(x_data2, a2, targets2):
    with tf.GradientTape() as tape:
        loss_value2 = loss2(x_data2, a2, targets2)
    return tape.gradient(loss_value2, [a2])

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Create variable (one model parameter = A)
A2 = tf.Variable(tf.random.normal(mean=10, shape=[1]))

# decide optimize method.
optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.02)

# Run loop
for i in range(2000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = tf.cast([x_vals_train[rand_index]], dtype=tf.float32)
    rand_y = tf.cast([y_vals_train[rand_index]], dtype=tf.float32)

    grands2 = grad2(rand_x, A, rand_y)

    optimizer2.apply_gradients(zip(grands2, [A]))
    if (i+1)%200==0:
        print('Step # {0} A = {1}'.format(i + 1, A.numpy()))
        tmploss = loss2(rand_x, A, rand_y)
        print('Loss = {0}'.format(tmploss))


# Evaluate Predictions on test set
y_prediction_train = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_vals_train, A))))
correct_prediction_train = tf.equal(y_prediction_train, y_vals_train)
accuracy_value_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))

y_prediction_test = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_vals_test, A))))
correct_prediction_test = tf.equal(y_prediction_test, y_vals_test)
accuracy_value_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

print('Accuracy on train set: {0}'.format(accuracy_value_train))
print('Accuracy on test set: {0}\n'.format(accuracy_value_test))

# Plot classification result
A_result = (-A).numpy()
bins = np.linspace(-5, 5, 50)
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='blue')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.axvline(A_result, ls = '--', linewidth=3, label='A = '+ str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(accuracy_value_test, 2)))
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         evaluating_models.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()