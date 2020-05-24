'''
*****************************************************************
# Back Propagation
#----------------------------------------------------------------
#
# This python function shows how to implement back propagation
# in regression and classification models.
****************************************************************
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import io
import datetime
from packaging import version
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(__doc__)

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '      Regression Example:                                                                                \n'
        '      We will create sample data as follows:                                                             \n'
        '      x-data: 100 random samples from a normal ~ N(1, 0.1)                                               \n'
        '      target: 100 values of the value 10.                                                                \n'
        '      We will fit the model:                                                                             \n'
        '      x-data * A = target                                                                                \n'
        '      Theoretically, A = 10.                                                                             \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(1., name='weight')
        self.B = tf.Variable(15., name='bias')
    def call(self, inputs):
        return inputs * self.W + self.B

# Create data
x_vals = np.random.normal(1, 0.1, 100)
noise = np.random.normal(0, 0.2, (100))
y_vals = np.repeat(10., 100) + noise

# Display raw data before training
# disply raw datas(3 * x + 2)
plt.figure(figsize=(8, 6))
plt.title("raw datas for y = 10")
plt.scatter(x_vals, y_vals, c='blue')
plt.grid(True)
plt.xlabel("x_vals")
plt.ylim(0, 20)
plt.ylabel("y_vals")
plt.show()

def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

# create model
model = Model()

# decide optimize method.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)

# calculate loss in initialize.
print("Initial loss: {:.3f}".format(loss(model, x_vals, y_vals)))

for i in range(800):
    grads = grad(model, x_vals, y_vals)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 50 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, x_vals, y_vals)))

print("Final loss: {:.3f}".format(loss(model, x_vals, y_vals)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

result_caluculate = model.W.numpy() * x_vals + model.B.numpy()

# disply raw datas(y = 10)
plt.figure(figsize=(8, 6))
plt.title("raw datas for y = 10")
plt.scatter(x_vals, y_vals, c='blue')
plt.plot(x_vals, result_caluculate, color='red')
plt.grid(True)
plt.ylim(0, 20)
plt.xlabel("x_vals")
plt.ylabel("y_vals")
plt.show()


print   (
        '---------------------------------------------------------------------------------------------------------\n'
        ' Classification Example                                                                                  \n'
        ' We will create sample data as follows:                                                                  \n'
        ' x-data: sample 50 random values from a normal = N(-1, 1)                                                \n'
        '        + sample 50 random values from a normal = N(3, 1)                                                \n'
        ' target: 50 values of 0 + 50 values of 1.                                                                \n'
        '         These are essentially 100 values of the corresponding output index                              \n'
        ' We will fit the binary classification model:                                                            \n'
        ' If sigmoid(x+A) < 0.5 -> 0 else 1                                                                       \n'
        ' Theoretically, A should be -(mean1 + mean2)/2                                                           \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )


# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50))) + noise

A = tf.Variable(tf.random.normal(shape=[1], mean=10, stddev=1.0, dtype=tf.dtypes.float32)).numpy()
my_output = tf.math.add(x_vals, A).numpy()

# histgram
# Display raw data before training
plt.figure(figsize=(10, 10))

plt.subplot(221)
plt.title("raw datas in x_vars")
plt.hist(x_vals)
plt.xlabel("x_vals")
plt.ylabel('Probability')

plt.subplot(222)
plt.title("raw datas in y_vars")
plt.hist(y_vals)
plt.xlabel("y_vals")
plt.ylabel('Probability')

plt.subplot(223)
plt.title("raw datas in A")
plt.hist(A)
plt.xlabel("A")
plt.ylabel('Probability')

plt.subplot(224)
plt.title("raw datas in my_output")
plt.hist(my_output)
plt.xlabel("my_output")
plt.ylabel('Probability')

plt.show()


# Display raw data before training
plt.figure(figsize=(8, 6))
plt.title("raw datas for y = 10")
plt.scatter(x_vals, my_output, c='blue')
plt.grid(True)
plt.xlabel("x_vals")
plt.ylim(-10, 20)
plt.ylabel("y_vals")
plt.show()

# create model
model = Model()

# decide optimize method.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

# calculate loss in initialize.
print("Initial loss: {:.3f}".format(loss(model, x_vals, my_output)))

for i in range(200):
    grads = grad(model, x_vals, my_output)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 50 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, x_vals, my_output)))

print("Final loss: {:.3f}".format(loss(model, x_vals, my_output)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

result_caluculate = model.W.numpy() * x_vals + model.B.numpy()

# disply raw datas(y = 10)
plt.figure(figsize=(8, 6))
plt.title("raw datas for y = 10")
plt.scatter(x_vals, my_output, c='blue')
plt.plot(x_vals, result_caluculate, color='red')
plt.grid(True)
plt.ylim(0, 20)
plt.xlabel("x_vals")
plt.ylabel("y_vals")
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         back_propagation_modify.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()