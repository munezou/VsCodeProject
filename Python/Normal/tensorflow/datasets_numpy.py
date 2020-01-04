'''
------------------------------------------------------------------------------------------
tf.data
    Numpy
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
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Loading from .npz files                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file(
        PROJECT_ROOT_DIR.joinpath('original_data/datasets/Mnist/mnist.npz'),
        DATA_URL
    )

with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Load a NumPy array using tf.data.Dataset.                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Suppose you have an array of samples and an array of corresponding labels. 
Input these two arrays to tf.data.Dataset.from_tensor_slices as tuples to create tf.data.Dataset.
---------------------------------------------------------------------------------------------------------------
'''
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

train_element_spec = train_dataset.element_spec
print('train_element_spec = \n{0}\n'.format(train_element_spec))

test_element_spec = test_dataset.element_spec
print('test_element_spec = \n{0}\n'.format(test_element_spec))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Using datasets                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Shuffle and batch datasets
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Model building and training                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile   (
                    optimizer=tf.keras.optimizers.RMSprop(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )

print('---< fitting model >---')
model_fit = model.fit(train_dataset, epochs=10)

print('---< Evaluate the model. >---')
model.evaluate(test_dataset)