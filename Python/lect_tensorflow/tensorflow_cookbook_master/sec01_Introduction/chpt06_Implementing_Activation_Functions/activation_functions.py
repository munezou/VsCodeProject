'''
***********************************************
# Activation Functions
#----------------------------------
#
# This function introduces activation
# functions in TensorFlow
**********************************************
'''
# Implementing Activation Functions
from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# X range
x_vals = np.linspace(start=-10., stop=10., num=100)

# ReLU activation
print('tf.nn.relu([-3., 3., 10.]) = {0}\n'.format(tf.nn.relu([-3., 3., 10.])))
y_relu = tf.nn.relu(x_vals)

# ReLU6 activation
print('tf.nn.relu6([-3., 3., 10.]) = {0}\n'.format(tf.nn.relu6([-3., 3., 10.])))
y_relu6 = tf.nn.relu6(x_vals)

# Sigmoid activation
print('tf.nn.sigmoid([-1., 0., 1.]) = {0}\n'.format(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = tf.nn.sigmoid(x_vals)

# Hyper Tangent activation
print('tf.nn.tanh([-1., 0., 1.]) = {0}\n'.format(tf.nn.tanh([-1., 0., 1.])))
y_tanh = tf.nn.tanh(x_vals)

# Softsign activation
print('tf.nn.softsign([-1., 0., 1.]) = {0}\n'.format(tf.nn.softsign([-1., 0., 1.])))
y_softsign = tf.nn.softsign(x_vals)

# Softplus activation
print('tf.nn.softplus([-1., 0., 1.]) = {0}\n'.format(tf.nn.softplus([-1., 0., 1.])))
y_softplus = tf.nn.softplus(x_vals)

# Exponential linear activation
print('tf.nn.elu([-1., 0., 1.]) = {0}\n'.format(tf.nn.elu([-1., 0., 1.])))
y_elu = tf.nn.elu(x_vals)

# Plot the different functions
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='upper left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='upper left')
plt.show()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         activation_function.py                                  ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()