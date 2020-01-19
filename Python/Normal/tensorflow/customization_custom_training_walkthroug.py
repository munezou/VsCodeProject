'''
------------------------------------------------------------------------------------------
customization
    Custom training: walkthrough

This guide uses machine learning to categorize Iris flowers by species. It uses TensorFlow to:

1. Build a model,
2. Train this model on example data, and
3. Use the model to make predictions about unknown data.
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
        '       TensorFlow programming                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This guide uses these high-level TensorFlow concepts:

* Use TensorFlow's default eager execution development environment,
* Import data with the Datasets API,
* Build models and layers with TensorFlow's Keras API.

This tutorial is structured like many TensorFlow programs:
1. Import and parse the dataset.
2. Select the type of model.
3. Train the model.
4. Evaluate the model's effectiveness.
5. Use the trained model to make predictions.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Configure imports                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Import TensorFlow and the other required Python modules. 
By default, TensorFlow uses eager execution to evaluate operations immediately, 
returning concrete values instead of creating a computational graph that is executed later. 
If you are used to a REPL or the python interactive console, this feels familiar.
---------------------------------------------------------------------------------------------------------------
'''
print("Eager execution: {}".format(tf.executing_eagerly()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The Iris classification problem                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Imagine you are a botanist seeking an automated way to categorize each Iris flower you find. 
Machine learning provides many algorithms to classify flowers statistically. 
For instance, a sophisticated machine learning program could classify flowers based on photographs. 
Our ambitions are more modest—we're going to classify Iris flowers based on the length 
and width measurements of their sepals and petals.

The Iris genus entails about 300 species, but our program will only classify the following three:

* Iris setosa
* Iris virginica
* Iris versicolor

Fortunately, someone has already created a dataset of 120 Iris flowers with the sepal and petal measurements. 
This is a classic dataset that is popular for beginner machine learning classification problems.
-----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Import and parse the training dataset                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Download the dataset file and convert it into a structure that can be used by this Python program.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the dataset                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Download the training dataset file using the tf.keras.utils.get_file function. 
This returns the file path of the downloaded file:
---------------------------------------------------------------------------------------------------------------
'''
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(
                                            origin=train_dataset_url,
                                            fname=PROJECT_ROOT_DIR.joinpath('csv_data/iris/iris_training.csv'), 
                                        )

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {0}".format(feature_names))
print("Label: {0}\n".format(label_name))

