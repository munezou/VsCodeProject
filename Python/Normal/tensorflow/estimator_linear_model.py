'''
------------------------------------------------------------------------------------------
estimator
    Build a linear model with Estimators

Overview
This end-to-end walkthrough trains a logistic regression model using the tf.estimator API. 
The model is often used as a baseline for other, more complex, algorithms.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import platform
import shutil
import subprocess
import random
from pathlib import Path
from packaging import version
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load the titanic dataset                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
You will use the Titanic dataset with the (rather morbid) goal of predicting passenger survival, given characteristics 
such as gender, age, class, etc.
--------------------------------------------------------------------------------------------------------------
'''

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       LExplore the data                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
The dataset contains the following features
---------------------------------------------------------------------------------------------------------------
'''

print('dftrain.head() = \n{0}\n'.format(dftrain.head()))

print('dftrain.describe() = \n{0}\n'.format(dftrain.describe()))

'''
--------------------------------------------------------------------------------------------------------------
There are 627 and 264 examples in the training and evaluation sets, respectively.
--------------------------------------------------------------------------------------------------------------
'''

print('(dftrain.shape[0] = {0}, dfeval.shape[0] = {1})\n'.format(dftrain.shape[0], dfeval.shape[0]))

# The majority of passengers are in their 20's and 30's.
dftrain.age.hist(bins=20)
plt.show()

# There are approximately twice as many male passengers as female passengers aboard.
dftrain.sex.value_counts().plot(kind='barh')
plt.show()

# The majority of passengers were in the "third" class.
dftrain['class'].value_counts().plot(kind='barh')
plt.show()

# Females have a much higher chance of surviving versus males. 
# This is clearly a predictive feature for the model.
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Feature Engineering for the Model                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Estimators use a system called feature columns to describe how the model should interpret each of the raw input features. 
An Estimator expects a vector of numeric inputs, and feature columns describe how the model should convert each feature.

Selecting and crafting the right set of feature columns is key to learning an effective model. 
A feature column can be either one of the raw inputs in the original features dict (a base feature column), 
or any new columns created using transformations defined over one or multiple base columns (a derived feature columns).

The linear estimator uses both numeric and categorical features. 
Feature columns work with all TensorFlow estimators and their purpose is to define the features used for modeling. 
Additionally, they provide some feature engineering capabilities like one-hot-encoding, normalization, and bucketization.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       FBase Feature Columns                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

'''
---------------------------------------------------------------------------------------------------------------
The input_function specifies how data is converted to a tf.data.Dataset that feeds the input pipeline in a streaming fashion. 
tf.data.Dataset take take in multiple sources such as a dataframe, a csv-formatted file, and more.
---------------------------------------------------------------------------------------------------------------
'''

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
            
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# You can inspect the dataset:
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys()))
    print()
    print('A batch of class:', feature_batch['class'].numpy())
    print()
    print('A batch of Labels:', label_batch.numpy())

# You can also inspect the result of a specific feature column using the tf.keras.layers.DenseFeatures layer:
age_column = feature_columns[7]
tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy()

'''
------------------------------------------------------------------------------------------------------------
DenseFeatures only accepts dense tensors, 
to inspect a categorical column you need to transform that to a indicator column first:
------------------------------------------------------------------------------------------------------------
'''
gender_column = feature_columns[0]
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()

# After adding all the base features to the model, let's train the model. 
# Training a model is just a single command using the tf.estimator API:
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Derived Feature Columns                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Now you reached an accuracy of 75%. 
Using each base feature column separately may not be enough to explain the data. 
For example, the correlation between gender and the label may be different for different gender. 
Therefore, if you only learn a single model weight for gender="Male" and gender="Female", 
you won't capture every age-gender combination 
(e.g. distinguishing between gender="Male" AND age="30" AND gender="Male" AND age="40").

To learn the differences between different feature combinations, 
you can add crossed feature columns to the model (you can also bucketize age column before the cross column):
---------------------------------------------------------------------------------------------------------------
'''

age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)

# After adding the combination feature to the model, let's train the model again:
derived_feature_columns = [age_x_gender]
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result)

'''
--------------------------------------------------------------------------------------------------------------
It now achieves an accuracy of 77.6%, which is slightly better than only trained in base features. 
You can try using more features and transformations to see if you can do better!

Now you can use the train model to make predictions on a passenger from the evaluation set. 
TensorFlow models are optimized to make predictions on a batch, or collection, of examples at once. 
Earlier, the eval_input_fn was defined using the entire evaluation set.
--------------------------------------------------------------------------------------------------------------
'''

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

'''
-------------------------------------------------------------------------------------------------------------
Finally, look at the receiver operating characteristic (ROC) of the results, 
which will give us a better idea of the tradeoff between the true positive rate and false positive rate.
-------------------------------------------------------------------------------------------------------------
'''

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        estimator_linear_model.py　　　　　       　  (2020/05/16)                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()