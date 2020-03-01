'''
------------------------------------------------------------------------------------------
structed data
    Classify structured data with feature columns

This tutorial demonstrates how to classify structured data (e.g. tabular data in a CSV). 
We will use Keras to define the model, 
and feature columns as a bridge to map from columns in a CSV to features used to train the model. 
This tutorial contains complete code to:

	* Load a CSV file using Pandas.
	* Build an input pipeline to batch and shuffle the rows using tf.data.
	* Map from columns in the CSV to features used to train the model using feature columns.
	* Build, train, and evaluate a model using Keras.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import feature_column

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

print   (
        '---------------------------------------------------------------------------------\n'
        '      The Dataset                                                                \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
We will use a small dataset provided by the Cleveland Clinic Foundation for Heart Disease. 
There are several hundred rows in the CSV. 
Each row describes a patient, and each column describes an attribute. 
We will use this information to predict whether a patient has heart disease, 
which in this dataset is a binary classification task.

Following is a description of this dataset. 
Notice there are both numeric and categorical columns.
-----------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/structed_datasets_00.jpg'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Use Pandas to create a dataframe                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Pandas is a Python library with many helpful utilities for loading and working with structured data. 
We will use Pandas to download the dataset from a URL, and load it into a dataframe.
-----------------------------------------------------------------------------------------
'''
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
print('dataframe.head() = \n{0}\n'.format(dataframe.head()))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Split the dataframe into train, validation, and test                       \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------
The dataset we downloaded was a single CSV file. 
We will split this into train, validation, and test sets.
-------------------------------------------------------------------------------------------
'''
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

print   (
        '---------------------------------------------------------------------------------\n'
        '      Create an input pipeline using tf.data                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Next, we will wrap the dataframes with tf.data. 
This will enable us to use feature columns as a bridge to map from the columns in the Pandas dataframe 
to features used to train the model. 
If we were working with a very large CSV file (so large that it does not fit into memory), 
we would use tf.data to read it from disk directly. That is not covered in this tutorial.
------------------------------------------------------------------------------------------
'''
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print('train_ds = \n{0}\n'.format(train_ds))
print('val_ds = \n{0}\n'.format(val_ds))
print('test_ds = \n{0}\n'.format(test_ds))