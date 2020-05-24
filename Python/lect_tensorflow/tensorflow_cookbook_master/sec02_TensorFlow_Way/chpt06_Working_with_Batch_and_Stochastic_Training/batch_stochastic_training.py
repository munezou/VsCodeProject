'''
# Batch and Stochastic Training
#----------------------------------
#
#  This python function illustrates two different training methods:
#  batch and stochastic training.  For each model, we will use
#  a regression model that predicts one model variable.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import io
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(__doc__)

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '                            Stochastic Training                                                          \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(1., name='weight')
        self.B = tf.Variable(15., name='bias')
    def call(self, inputs):
        return inputs * self.W + self.B

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '                            Generate Data                                                                \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )
'''
The data we will create is 100 random samples from a Normal(mean = 1, sd = 0.1). 
The target will be an array of size 100 filled with the constant 10.0.

We also create the necessary placeholders in the graph for the data and target. 
Note that we use a shape of [1] for stochastic training.
'''
x_vals = np.random.normal(1, 0.1, 100)
noise = np.random.normal(0, 0.2, (100))
y_vals = np.repeat(10., 100) + noise

# Display raw data before training
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

loss_stochastic = []

# training
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    grads = grad(model, rand_x, rand_y)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 5 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, rand_x, rand_y)))
        loss_stochastic.append(loss(model, rand_x, rand_y))

# Display the training result
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
        '          Batch Training                                                                                 \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )
'''
For Batch training, we need to declare our batch size. 
The larger the batch size, the smoother the convergence will be towards the optimal value. 
But if the batch size is too large, the optimization algorithm may get stuck in a local minimum, 
where a more stochastic convergence may jump out.

Here, the we may change the batch size from 1 to 100 to see the effects of the batch size on the convergence plots at the end.
'''

# Declare batch size
batch_size = 25

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '          Generate the Data                                                                              \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )
'''
The data we will create is 100 random samples from a Normal(mean = 1, sd = 0.1). 
The target will be an array of size 100 filled with the constant 10.0.

We also create the necessary placeholders in the graph for the data and target.

Note that here, our placeholders have shape [None, 1], where the batch size will take the place of the None dimension.
'''

# Create data
x_vals = np.random.normal(1, 0.1, 100)
noise = np.random.normal(0, 0.2, (100))
y_vals = np.repeat(10., 100) + noise

# Display raw data before training
plt.figure(figsize=(8, 6))
plt.title("raw datas for y = 10")
plt.scatter(x_vals, y_vals, c='blue')
plt.grid(True)
plt.xlabel("x_vals")
plt.ylim(0, 20)
plt.ylabel("y_vals")
plt.show()

# create model
model = Model()

# decide optimize method.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

# calculate loss in initialize.
print("Initial loss: {:.3f}".format(loss(model, x_vals, y_vals)))

loss_batch = []

# training
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])

    grads = grad(model, rand_x, rand_y)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 5 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, rand_x, rand_y)))
        loss_batch.append(loss(model, rand_x, rand_y))

# Display the training result
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
        '          Plot Stochastic vs Batch Training                                                              \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         batch_stochastic_training.py                          ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()
