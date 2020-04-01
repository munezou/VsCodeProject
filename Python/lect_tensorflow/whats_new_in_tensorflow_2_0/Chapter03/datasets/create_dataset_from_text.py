'''
------------------------------------------------------------------------------------------
chapter03/datasets
    create_dataset_from_text
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
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '                         create_dataset_from_text                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

def train_decode_line(row):
    cols = tf.io.decode_csv(row, record_defaults=[[0.], ['house'], [0.]])
    myfeatures = {'sq_footage':cols[0], 'type':cols[1]}
    mylabel = cols[2] #price
    return myfeatures, mylabel

def predict_decode_line(row):
    cols = tf.decode_csv(row, record_defaults=[[0.], ['house']])
    myfeatures = {'sq_footage':cols[0], 'type':cols[1]}
    return myfeatures

line_dataset = tf.data.TextLineDataset(os.path.join(PROJECT_ROOT_DIR, "curated_data/train.csv"))

print('line_dataset = \n{0}\n'.format(line_dataset))

for line in line_dataset.take(4):
    print(line)

print()

train_dataset = line_dataset.map(train_decode_line)

for train in train_dataset.take(4):
    print(train)