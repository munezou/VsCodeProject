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

print   (
        '---------------------------------------------------------------------------------\n'
        '      Understand the input pipeline                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Now that we have created the input pipeline, let's call it to see the format of the data it returns. 
We have used a small batch size to keep the output readable.
----------------------------------------------------------------------------------------
'''
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )

'''
----------------------------------------------------------------------------------------
We can see that the dataset returns a dictionary of column names (from the dataframe) 
that map to column values from rows in the dataframe.
----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Demonstrate several types of feature column                                \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
TensorFlow provides many types of feature columns. 
In this section, we will create several types of feature columns, 
and demonstrate how they transform a column from the dataframe.
-----------------------------------------------------------------------------------------
'''
# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]

# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())

print   (
        '---------------------------------------------------------------------------------\n'
        '      Numeric columns                                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
The output of a feature column becomes the input to the model (using the demo function defined above, 
we will be able to see exactly how each column from the dataframe is transformed). 
A numeric column is the simplest type of column. 
It is used to represent real valued features. 
When using this column, 
your model will receive the column value from the dataframe unchanged.
------------------------------------------------------------------------------------------
'''
age = tf.feature_column.numeric_column("age")

demo(age)

'''
------------------------------------------------------------------------------------------
In the heart disease dataset, most columns from the dataframe are numeric.
------------------------------------------------------------------------------------------
'''

print   (
        '---------------------------------------------------------------------------------\n'
        '      Bucketized columns                                                         \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Often, 
you don't want to feed a number directly into the model, 
but instead split its value into different categories based on numerical ranges. 
Consider raw data that represents a person's age. 
Instead of representing age as a numeric column, 
we could split the age into several buckets using a bucketized column. 
Notice the one-hot values below describe which age range each row matches.
-----------------------------------------------------------------------------------------
'''
age_buckets = tf.feature_column.bucketized_column(
                    age, 
                    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
                )

demo(age_buckets)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Categorical columns                                                        \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
In this dataset, 
thal is represented as a string (e.g. 'fixed', 'normal', or 'reversible'). 
We cannot feed strings directly to a model. 
Instead, we must first map them to numeric values. 
The categorical vocabulary columns provide a way to represent strings as a one-hot vector 
(much like you have seen above with age buckets). 
The vocabulary can be passed as a list using categorical_column_with_vocabulary_list, 
or loaded from a file using categorical_column_with_vocabulary_file.
----------------------------------------------------------------------------------------
'''
thal = tf.feature_column.categorical_column_with_vocabulary_list(
            'thal', ['fixed', 'normal', 'reversible']
        )

thal_one_hot = tf.feature_column.indicator_column(thal)

demo(thal_one_hot)

'''
----------------------------------------------------------------------------------------
In a more complex dataset, many columns would be categorical (e.g. strings). 
Feature columns are most valuable when working with categorical data. 
Although there is only one categorical column in this dataset, 
we will use it to demonstrate several important types of feature columns 
that you could use when working with other datasets.
---------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Embedding columns                                                          \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------
Suppose instead of having just a few possible strings, 
we have thousands (or more) values per category. 
For a number of reasons, 
as the number of categories grow large, 
it becomes infeasible to train a neural network using one-hot encodings. 
We can use an embedding column to overcome this limitation. 
Instead of representing the data as a one-hot vector of many dimensions, 
an embedding column represents that data as a lower-dimensional, 
dense vector in which each cell can contain any number, not just 0 or 1. 
The size of the embedding (8, in the example below) is a parameter that must be tuned.

Key point: 
using an embedding column is best when a categorical column has many possible values. 
We are using one here for demonstration purposes, 
so you have a complete example you can modify for a different dataset in the future.
---------------------------------------------------------------------------------------
'''
# Notice the input to the embedding column is the categorical column
# we previously created
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)

demo(thal_embedding)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Hashed feature columns                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------
Another way to represent a categorical column with a large number of values is to use a categorical_column_with_hash_bucket. 
This feature column calculates a hash value of the input, 
then selects one of the hash_bucket_size buckets to encode a string. 
When using this column, 
you do not need to provide the vocabulary, 
and you can choose to make the number of hash_buckets significantly smaller than the number of actual categories to save space.

Key point: 
An important downside of this technique is that there may be collisions in which different strings are mapped to the same bucket. 
In practice, this can work well for some datasets regardless.
---------------------------------------------------------------------------------------
'''
thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
                'thal', hash_bucket_size=1000
            )

demo(tf.feature_column.indicator_column(thal_hashed))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Crossed feature columns                                                    \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Combining features into a single feature, 
better known as feature crosses, enables a model to learn separate weights for each combination of features. 
Here, we will create a new feature that is the cross of age and thal. 
Note that crossed_column does not build the full table of all possible combinations (which could be very large). 
Instead, it is backed by a hashed_column, so you can choose how large the table is.
-----------------------------------------------------------------------------------------
'''
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)

#demo(tf.feature_column.indicator_column(crossed_feature))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Choose which columns to use                                                \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
We have seen how to use several types of feature columns. 
Now we will use them to train a model. 
The goal of this tutorial is 
to show you the complete code (e.g. mechanics) needed to work with feature columns. 
We have selected a few columns to train our model below arbitrarily.

Key point: 
If your aim is to build an accurate model, try a larger dataset of your own, 
and think carefully about which features are the most meaningful to include, 
and how they should be represented.
------------------------------------------------------------------------------------------
'''
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age_buckets = tf.feature_column.bucketized_column(
                age, 
                boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
            )

feature_columns.append(age_buckets)

# indicator cols
thal = tf.feature_column.categorical_column_with_vocabulary_list(
            'thal', 
            ['fixed', 'normal', 'reversible']
        )

thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = tf.feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Create a feature layer                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Now that we have defined our feature columns, 
we will use a DenseFeatures layer to input them to our Keras model.
----------------------------------------------------------------------------------------
'''
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

'''
----------------------------------------------------------------------------------------
Earlier, we used a small batch size to demonstrate how feature columns worked. 
We create a new input pipeline with a larger batch size.
---------------------------------------------------------------------------------------
'''
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

print('train_ds = \n{0}\n'.format(train_ds))
print('val_ds = \n{0}\n'.format(val_ds))
print('test_ds = \n{0}\n'.format(test_ds))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Create, compile, and train the model                                       \n'
        '---------------------------------------------------------------------------------\n'
        )
model = tf.keras.Sequential([
            feature_layer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

'''
-------------------------------------------------------------------------------
Key point: 
You will typically see best results with deep learning with much larger and more complex datasets. 
When working with a small dataset like this one, 
we recommend using a decision tree or random forest as a strong baseline. 
The goal of this tutorial is not to train an accurate model, 
but to demonstrate the mechanics of working with structured data, 
so you have code to use as a starting point when working with your own datasets in the future.
------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        struct_feature_column.py                     　                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()