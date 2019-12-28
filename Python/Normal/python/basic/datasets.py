'''
------------------------------------------------------------------------------------------
tf.data

------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import platform
import shutil
import subprocess
from pathlib import Path
from packaging import version
from PIL import Image
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/python/basic')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dataset structure                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
tf_random_uniform = tf.random.uniform([4, 10])

for data in tf_random_uniform:
    print(data.numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf_random_uniform)
print('datasets1.element_spec = {0}\n'.format(dataset1.element_spec))

dataset2 = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([4]),
        tf.random.uniform([4, 100], 
        maxval=100, 
        dtype=tf.int32))
    )
print('datasets2.element_spec = {0}\n'.format(dataset2.element_spec))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print('dataset3.element_spec = {0}\n'.format(dataset3))

dataset = tf.data.Dataset.from_tensor_slices(
        {"a": tf.random.uniform([4]),
        "b": tf.random.uniform([4, 100], 
        maxval=100, 
        dtype=tf.int32)}
    )
print('dataset.element_spec = {0}\n'.format(dataset.element_spec))

print('---< map >---')
dataset1 = dataset1.map(lambda x: x + 1)
for data in dataset1:
    print(data.numpy())
    
print()
print('---< flat_map >---')
dataset4 = tf.data.Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
dataset4 = dataset4.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x + 1))
for data in dataset4:
    print(data.numpy())
print()

print('---< filter >---')
d = tf.data.Dataset.from_tensor_slices([1, 2, 3])

d = d.filter(lambda x: x < 3)  # ==> [1, 2]
print(list(d))

for data in d:
    print(data.numpy())

print()
# `tf.math.equal(x, y)` is required for equality comparison
def filter_fn(x):
    return tf.math.equal(x, 1)

d = d.filter(filter_fn)  # ==> [1]
for data in d:
    print(data.numpy())
print()

