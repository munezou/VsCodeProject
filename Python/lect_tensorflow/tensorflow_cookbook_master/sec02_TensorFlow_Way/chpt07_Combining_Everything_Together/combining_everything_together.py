'''
# Combining Everything Together
#----------------------------------
# This file will perform binary classification on the
# iris dataset. We will only predict if a flower is
# I.setosa or not.
#
# We will create a simple binary classifier by creating a line
# and running everything through a sigmoid to get a binary predictor.
# The two features we will use are pedal length and pedal width.
#
# We will use batch training, but this can be easily
# adapted to stochastic training.
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
from sklearn import datasets
import tensorflow as tf

print(__doc__)

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print(
        '---------------------------------------------------------------------------------------------------------\n'
        '                            Combining Everything Together                                                \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )
'''
This file will perform binary classification on the iris dataset. 
We will only predict if a flower is I.setosa or not.

We will create a simple binary classifier by creating a line and running everything through a sigmoid to get a binary predictor. 
The two features we will use are pedal length and pedal width. 
We use these two features because we know that Iris setosa is separable by these two features. 
We aim to find the line that separates it out.

We will use batch training, but this can be easily adapted to stochastic training (i.e. set batch size equal to 1).

We start by loading the necessary libraries and resetting the computational graph.
'''



# Load the iris data
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ [0:sepal.width, 1:sepal.length, 2:pedal.width, 3:pedal.length]
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '                            Model Variables                                                              \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )

# Create variables A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))

print   (
        '---------------------------------------------------------------------------------------------------------\n'
        '                            Model Operations                                                             \n'
        '---------------------------------------------------------------------------------------------------------\n'
    )
'''
A line can be defined as  𝑥1=𝐴⋅𝑥2+𝑏 . 
To create a linear separator, we would like to see which side of the line the data points fall. 
There are three cases:

	* A point exactly on the line will satisfy:  0=𝑥1−(𝐴⋅𝑥2+𝑏) 
	* A point above the line satisfies:  0>𝑥1−(𝐴⋅𝑥2+𝑏) 
	* A point below the line satisfies:  0<𝑥1−(𝐴⋅𝑥2+𝑏) 
We will make the output of this model:

	𝑥1−(𝐴⋅𝑥2+𝑏)

Then the predictions will be the sign of that output:

	𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛(𝑥1,𝑥2)=𝑠𝑖𝑔𝑛(𝑥1−(𝐴⋅𝑥2+𝑏))

So we add the corresponding operations to the computational graph.
'''
batch_size = 20

def loss(aa, bb, input_1, input_2, targets):
    my_mult = tf.matmul(input_2, aa)
    my_add = tf.add(my_mult, bb)
    my_output = tf.subtract(input_1, my_add)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=targets))

def grad(aa, bb, input_1, input_2, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(aa, bb, input_1, input_2, targets)
    return tape.gradient(loss_value, [aa, bb])

# decide optimize method.
optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

loss_batch = []

# Run Loop
for i in range(8000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = tf.cast(np.array([[x[0]] for x in rand_x]), dtype=tf.float32)
    rand_x2 = tf.cast(np.array([[x[1]] for x in rand_x]), dtype=tf.float32)
    rand_y = tf.cast(np.array([[y] for y in binary_target[rand_index]]), dtype=tf.float32)
    grands = grad(A, b, rand_x1, rand_x2, rand_y)

    optimizer.apply_gradients(zip(grands, [A, b]))
    if (i+1)%200==0:
        tmploss = loss(A, b, rand_x1, rand_x2, rand_y)
        print('Step #{0} loss = {1}, A = {2}, b = {3}'.format(i + 1, tmploss, A.numpy(), b.numpy()))
        loss_batch.append(tmploss)

# Extract the coefficients.
[[slope]] = A.numpy()
[[intercept]] = b.numpy()

# Create a straight line that fits.
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope*i+intercept)

# Plot the fitted line over the data
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()


date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         back_propagation.py                             ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()