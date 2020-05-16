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
import random
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

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python' ,'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
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

print('---< map >---')
'''
-----------------------------------------------------------------------------------------------------------------
map(
    map_func,
    num_parallel_calls=None
)

Maps map_func across the elements of this dataset.

This transformation applies map_func to each element of this dataset, 
and returns a new dataset containing the transformed elements, 
in the same order as they appeared in the input.
------------------------------------------------------------------------------------------------------------------
'''
a = tf.data.Dataset.range(1, 6)
a1 = a.map(lambda x : x + 1)
for data in a1:
    print(data.numpy())
    
# The input signature of map_func is determined by the structure of each element in this dataset. For example:
# NOTE: The following examples use `{ ... }` to represent the
# contents of a dataset.
# Each element is a `tf.Tensor` object.
a2 = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])
print('a = {0}, '.format(a2))

# `map_func` takes a single argument of type `tf.Tensor` with the same
# shape and dtype.
a2 = a2.map(lambda x: tf.math.sqrt(x))
print('a2.map(lambda x: tf.math.sqrt(x)) = \n{0}\n'.format(a2))

for data in a2:
    print(data.numpy())

'''
# Each element is a tuple containing two `tf.Tensor` objects.
b = tf.data.Dataset(tf.tuple((1, "foo"), (2, "bar"), (3, "baz")))
print('b = \n{0}\n'.format(b))

for data in b:
    print(data)

# `map_func` takes two arguments of type `tf.Tensor`.
result = b.map(lambda x_int, y_str: (x_int + 1, y_str))
print('b.map(lambda x_int, y_str: ...) = \n{0}\n'.format(result))


# Each element is a dictionary mapping strings to `tf.Tensor` objects.
c = { {"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"} }
print('c = \n{0}\n'.format(c))

# `map_func` takes a single argument of type `dict` with the same keys as
# the elements.
result = c.map(lambda d: ...)
print('c.map(lambda d: ...) = \n{0}\n'.format(result))
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        datasets.py                               (2020/05/16)                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()