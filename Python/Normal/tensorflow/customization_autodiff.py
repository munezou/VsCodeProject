'''
------------------------------------------------------------------------------------------
customization
    Automatic differentiation and gradient tape

In the previous tutorial we introduced Tensors and operations on them. 
In this tutorial we will cover automatic differentiation, 
a key technique for optimizing machine learning models.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
from pathlib import Path
from packaging import version
from PIL import Image
import tempfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Gradient tapes                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
TensorFlow provides the tf.GradientTape API for automatic differentiation - computing the gradient of a computation 
with respect to its input variables. 
Tensorflow "records" all operations executed inside the context of a tf.GradientTape onto a "tape". 
Tensorflow then uses that tape and the gradients associated with each recorded operation to compute the gradients of a "recorded" computation 
using reverse mode differentiation.
----------------------------------------------------------------------------------------------------------------
'''

# For example:
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

'''
-----------------------------------------------------------------------------------------------------------------
You can also request gradients of the output with respect 
to intermediate values computed during a "recorded" tf.GradientTape context.
-----------------------------------------------------------------------------------------------------------------
'''
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

'''
------------------------------------------------------------------------------------------------------------------
By default, 
the resources held by a GradientTape are released as soon as GradientTape.gradient() method is called. 
To compute multiple gradients over the same computation, create a persistent gradient tape. 
This allows multiple calls to the gradient() method as resources are released when the tape object is garbage collected. 
For example:
------------------------------------------------------------------------------------------------------------------
'''
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Recording control flow                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Because tapes record operations as they are executed, 
Python control flow (using ifs and whiles for example) is naturally handled:
--------------------------------------------------------------------------------------------------------------
'''

def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Higher-order gradients                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Operations inside of the GradientTape context manager are recorded for automatic differentiation. 
If gradients are computed in that context, then the gradient computation is recorded as well. As a result, 
the exact same API works for higher-order gradients as well. For example:
---------------------------------------------------------------------------------------------------------------
'''
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)

d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished         customization_autodiff.py                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()