'''
------------------------------------------------------------------------------------------
tf.data
    csv
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
        '       Load csv file using tf.data.                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file(
        origin=TRAIN_DATA_URL,
        fname=PROJECT_ROOT_DIR.joinpath('original_data/titanic/train.csv')
    )


test_file_path = tf.keras.utils.get_file(
        origin=TEST_DATA_URL,
        fname=PROJECT_ROOT_DIR.joinpath('original_data/titanic/eval.csv')
    )

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# output first row
with open(train_file_path, 'r') as f:
    names_row = f.readline()

CSV_COLUMNS = names_row.rstrip('\n').split(',')
print('CSV_COLUMNS = \n{0}\n'.format(CSV_COLUMNS))

# You need to identify the column that will be the label for each sample and indicate what it is.
LABELS = [0, 1]
LABEL_COLUMN = 'survived'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=12, # It is set small to make it easier to see.
    label_name=LABEL_COLUMN,
    na_value="?",
    num_epochs=1,
    ignore_errors=True)
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

'''
-----------------------------------------------------------------------------------------------------
The elements that make up the dataset are batches represented as tuples of the form 
(multiple samples, multiple labels). 
The data in the sample is organized as column-based tensors (rather than row-based tensors), 
each containing a batch-sized (12 in this case) component.
-----------------------------------------------------------------------------------------------------
'''
examples, labels = next(iter(raw_train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Preprocessing of data.                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------------
Category data
Some columns in this CSV data are category columns. 
That is, its content must be one of a limited set of options.

In this CSV, these choices are represented as text. 
This text needs to be converted to numbers so that you can train the model. 
To make this easier, you need to create a list of category columns and a list of their choices.
---------------------------------------------------------------------------------------------------------
'''
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

'''
----------------------------------------------------------------------------------------------------------
Write a function that takes a categorical value tensor, matches it with a list of value names, 
and then performs one-hot encoding.
----------------------------------------------------------------------------------------------------------
'''
def process_categorical_data(data, categories):
    """
    ------------------------------------------------------------------
    Returns a one-hot encoded tensor representing category values.
    ------------------------------------------------------------------
    """
    
    # Remove the first ' '
    data = tf.strings.regex_replace(data, '^ ', '')
    # Remove the last '.'
    data = tf.strings.regex_replace(data, r'\.$', '')
    
    # One-hot encoding
    # Reshape data from one dimension (list) to two dimensions (list of one element list).
    data = tf.reshape(data, [-1, 1])
    
    # For each element, create a list of the boolean values of the number of categories, 
    # where the label of the element and the category match are True.
    data = categories == data
    
    # Casts a boolean to a floating point number.
    data = tf.cast(data, tf.float32)
    
    # You can also put the entire encoding on one line:
    # data = tf.cast(categories == tf.reshape(data, [-1, 1]), tf.float32)
    return data

'''
-----------------------------------------------------------------------
To visualize this process, 
we take one tensor of the category column from the first batch, process it, 
and show the state before and after.
------------------------------------------------------------------------
'''
class_tensor = examples['class']
print('class_tensor = \n{0}\n'.format(class_tensor))

class_categories = CATEGORIES['class']
print('class_categories = {0}\n'.format(class_categories))

processed_class = process_categorical_data(class_tensor, class_categories)
print('processed_class = \n{0}\n'.format(processed_class))

'''
----------------------------------------------------------------------------
Notice the relationship between the length of the two inputs and the shape of the output.
----------------------------------------------------------------------------
'''
print('Size of batch: {0}'.format(len(class_tensor.numpy())))
print('Number of category labels: {0}'.format(len(class_categories)))
print('Shape of one-hot encoded tensor: {0}\n'.format(processed_class.shape))

'''
----------------------------------------------------------------------------
Continuous data
Continuous data must be normalized so that the value is between 0 and 1. 
To do this, 
write a function that multiplies each value by 1 divided by twice the average of the column values.
----------------------------------------------------------------------------
'''

# This function also reshapes the data into a two-dimensional tensor.
def process_continuous_data(data, mean):
    # standardization of data
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])

'''
---------------------------------------------------------------------------
To perform this calculation, you need the average of the column values. 
Obviously, in reality it is necessary to calculate this value, 
but we will show a value for this example.
---------------------------------------------------------------------------
'''
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}


'''
-----------------------------------------------------------------------------
To perform this calculation, you need the average of the column values. 
Obviously, in reality it is necessary to calculate this value, 
but we will show a value for this example.
-----------------------------------------------------------------------------
'''
# To see what this function actually does, 
# take a continuous tensor and look before and after.
age_tensor = examples['age']
print('age_tensor = \n{0}\n'.format(age_tensor))

age_tensor_standard = process_continuous_data(age_tensor, MEANS['age'])
print('age_tensor_standard = \n{0}\n'.format(age_tensor_standard))

'''
------------------------------------------------------------------------------
Data preprocessing

Combine these preprocessing tasks into a single function 
that can be mapped to batches in a dataset.
------------------------------------------------------------------------------
'''
def preprocess(features, labels):
        
    # Processing category features
    for feature in CATEGORIES.keys():
        features[feature] = process_categorical_data(features[feature], CATEGORIES[feature])

    # Processing of continuous features
    for feature in MEANS.keys():
        features[feature] = process_continuous_data(features[feature], MEANS[feature])
    
    # Assemble features into one tensor.
    features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)
    
    return features, labels

# Then apply it using the tf.Dataset.map function and shuffle the dataset to prevent overtraining.
train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)

# Let's see what one sample looks like.
examples, labels = next(iter(train_data))

print('examples = \n{0}\n'.format(examples))
print('labels = \n{0}\n'.format(labels))

'''
----------------------------------------------------------------------------------
This example consists of a two-dimensional array with 12 items (the batch size). 
Each item represents one line of the original CSV file. 
Labels are one-dimensional tensors with 12 values.
-----------------------------------------------------------------------------------
'''

'''
-----------------------------------------------------------------------------------
Build the model

In this example, 
we use the Keras Functional API and wrap it with the get_model constructor to build a simple model.
-----------------------------------------------------------------------------------
'''

def get_model(input_dim, hidden_units=[100]):
    """
    -----------------------------------------------------
    Create Keras model with multiple layers

    argument:
        input_dim: (int) shape of items in batch
        labels_dim: (int) label shape
        hidden_units: [int] Layer size of DNN (input layer first)
        learning_rate: (float) optimizer learning rate

    Return value:
        Keras model
    ------------------------------------------------------
    """

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

'''
--------------------------------------------------------------------
The get_model constructor needs 
to know the shape of the input data (except the batch size).
--------------------------------------------------------------------
'''
train_element_spec = train_data.element_spec

input_shape = train_element_spec[0]
output_shape = train_element_spec[1] # [0] is the batch size

input_dimension = input_shape.shape.dims[1]

print('(input_shape = {0}, output_shape = {1})\n'.format(input_shape, output_shape))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Training, evaluation, and prediction                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
model = get_model(input_dimension)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_data, epochs=20)

# Once you have trained your model, you can check the accuracy rate on the test_data dataset.
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {0}, Test Accuracy {1}\n'.format(test_loss, test_accuracy))

# Use tf.keras.Model.predict to infer the labels of a single batch or a dataset consisting of batches.
predictions = model.predict(test_data)

# View some of the results.
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
    " | Actual outcome: ",
    ("SURVIVED" if bool(survived) else "DIED"))