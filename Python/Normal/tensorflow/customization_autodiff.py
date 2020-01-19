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
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
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

