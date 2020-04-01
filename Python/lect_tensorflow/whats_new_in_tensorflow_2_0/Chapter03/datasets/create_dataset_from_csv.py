'''
------------------------------------------------------------------------------------------
chapter03/datasets
    Example to show creation of tf.data.Dataset using csv
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
        ' Read a csv file having three columns. We will name them to 'square_ft', 'house_type', and 'price'    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

csv_file = os.path.join(PROJECT_ROOT_DIR, "curated_data/train.csv")
csv_columns = ['square_ft', 'house_type', 'price']

dataset = tf.data.experimental.make_csv_dataset(csv_file, column_names=csv_columns, batch_size=8)

print('dataset = \n{0}\n'.format(dataset))

for data in dataset.take(4):
    print(data)