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
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load a pandas DataFrame using tf.data.                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
contents)

This tutorial shows an example of loading a pandas DataFrame and loading the data into tf.data.Dataset.

This tutorial uses a small dataset provided by the Cleveland Clinic Foundation for Heart Disease. 
This dataset (CSV) contains hundreds of rows of data. 
The rows represent each patient and the columns represent various attributes.

This data can be used to determine and predict whether a patient has heart disease. 
This is a binary classification problem.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Read data using pandas.                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Download the CSV containing the heart dataset.
DATA_URL = 'https://storage.googleapis.com/applied-dl/heart.csv'

csv_file = tf.keras.utils.get_file(
        PROJECT_ROOT_DIR.joinpath('original_data/datasets/Heart/heart.csv'),
        DATA_URL
    )

# Read the csv using pandas
df = pd.read_csv(csv_file)

print('df.head() = \n{0}\n'.format(df.head()))

print('df.dtypes = \n{0}\n'.format(df.dtypes))

# Converts the only object type thal column in a dataframe to a discrete value.
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print('df.head() = \n{0}\n'.format(df.head()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load data using tf.data.Dataset.                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# Display the first 5 lines of tf.data.Dataset.
for feat, targ in dataset.take(5):
    print ('Features: {0}, Target: {1}'.format(feat, targ))

'''
-------------------------------------------------------------------------------------------------------------
pd.Series implements the __array__ protocol, so it can be used almost anywhere you use np.array or tf.Tensor.
-------------------------------------------------------------------------------------------------------------
'''
tf_constant_df = tf.constant(df['thal'])
print('tf.constant(df["thal"]) = \n{0}\n'.format(tf_constant_df))

# Performs batch processing by shuffling data.
train_dataset = dataset.shuffle(len(df)).batch(1)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create and train a model.                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# create model
model = get_compiled_model()

print('---< fitting model >---')
model.fit(train_dataset, epochs=15)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Alternate feature columns                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Passing dictionary data to the input to the model is as easy as creating a dictionary of the same type in tf.keras.layers.Input, 
applying some preprocessing, and stacking using functional api. 
Can be done. 
This can be used instead of the feature column.
---------------------------------------------------------------------------------------------------------------
'''
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

'''
-------------------------------------------------------------------------------------------------------------------
The easiest way to preserve the column structure of a pandas DataFrame 
when using tf.data is to convert the DataFrame to dictionary data and truncate it.
-------------------------------------------------------------------------------------------------------------------
'''
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print ('dict_slice = \n{0}\n'.format(dict_slice))

model_func.fit(dict_slices, epochs=15)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        datasets_pandas_DataFrame.py                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()