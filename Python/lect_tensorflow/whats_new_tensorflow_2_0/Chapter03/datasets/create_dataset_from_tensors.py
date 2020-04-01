'''
------------------------------------------------------------------------------------------
chapter03/datasets
    Illustration of tf.data.Dataset.from_tensors(...) and tf.data.Dataset.from_tensor_slices(...) APIs
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
from packaging import version
from PIL import Image
import itertools

import numpy as np
import pandas as pd

import tensorflow as tf


print(__doc__)

# Display current path
# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        ' Illustration of tf.data.Dataset.from_tensors(...) and tf.data.Dataset.from_tensor_slices(...) APIs   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

aa = np.array([1, 2, 3])
bb = np.array([2, 3, 4])
cc = np.array([3, 4, 5])
dd = np.array([4, 5, 6])

features = [aa, bb, cc, dd]

print('features = \n{0}\n'.format(features))

labels = ['aa', 'bb', 'cc', 'dd']

print('labels = {0}\n'.format(labels))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        'tf.data.Dataset.from_tensors(...) creates a Dataset with a single element comprising the given tensors\n'
        '------------------------------------------------------------------------------------------------------\n'
        )

dataset1 = tf.data.Dataset.from_tensors((features, labels))

print('dataset1 = \n{0}\n'.format(dataset1))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Iterate dataset1                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

for data in dataset1:
    print(data)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      tf.data.Dataset.from_tensors_slices(...) creates a Dataset                                      \n' 
        '                           whose elements are slices from the given tensor(s)                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

dataset2 = tf.data.Dataset.from_tensor_slices((features, labels))

print('dataset2 = \n{0}\n'.format(dataset2))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Iterate dataset2                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

for data in dataset2:
    print(data)
