'''
------------------------------------------------------------------------------------------
estimator
    Boosted trees using Estimators

This tutorial is an end-to-end walkthrough of training a Gradient Boosting model using decision trees with the tf.estimator API. 
Boosted Trees models are among the most popular and effective machine learning approaches for both regression and classification. 
It is an ensemble technique that combines the predictions from several (think 10s, 100s or even 1000s) tree models.

Boosted Trees models are popular with many machine learning practitioners 
as they can achieve impressive performance with minimal hyperparameter tuning.
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

import tensorflow.compat.v2.feature_column as fc
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
---------------------------------------------------------------------------------------------------------------
You will be using the titanic dataset, 
where the (rather morbid) goal is to predict passenger survival, given characteristics such as gender, age, class, etc.
---------------------------------------------------------------------------------------------------------------
'''
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

tf.random.set_seed(123)

'''
---------------------------------------------------------------------------------------------------------------
The dataset consists of a training set and an evaluation set:

dftrain and y_train are the training set—the data the model uses to learn.
The model is tested against the eval set, dfeval, and y_eval.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Explore the data                                                                               \n'
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
        '       Create feature columns and input functions                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
The Gradient Boosting estimator can utilize both numeric and categorical features. 
Feature columns work with all TensorFlow estimators and their purpose is to define the features used for modeling. 
Additionally they provide some feature engineering capabilities like one-hot-encoding, normalization, and bucketization. 
In this tutorial, the fields in CATEGORICAL_COLUMNS are transformed from categorical columns to one-hot-encoded columns
(indicator column):
---------------------------------------------------------------------------------------------------------------
'''

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab)
    )
    
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

'''
--------------------------------------------------------------------------------------------------------------
You can view the transformation that a feature column produces. For example, 
here is the output when using the indicator_column on a single example:
--------------------------------------------------------------------------------------------------------------
'''

example = dict(dftrain.head(1))

class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))

print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())
print()

'''
---------------------------------------------------------------------------------------------------------------
Additionally, you can view all of the feature column transformations together:
---------------------------------------------------------------------------------------------------------------
'''

tf_keras_layers_DenseFeatures_example = tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()

print('tf_keras_layers_DenseFeatures_example = \n{0}\n'.format(tf_keras_layers_DenseFeatures_example))

'''
--------------------------------------------------------------------------------------------------------------
Next you need to create the input functions. 
These will specify how data will be read into our model for both training and inference. 
You will use the from_tensor_slices method in the tf.data API to read in data directly from Pandas. 
This is suitable for smaller, in-memory datasets. 
For larger datasets, the tf.data API supports a variety of file formats (including csv) 
so that you can process datasets that do not fit in memory.
--------------------------------------------------------------------------------------------------------------
'''

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    
    return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train and evaluate the model                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Below you will do the following steps:

1. Initialize the model, specifying the features and hyperparameters.
2. Feed the training data to the model using the train_input_fn and train the model using the train function.
3. You will assess model performance using the evaluation set—in this example, the dfeval DataFrame. 
    You will verify that the predictions match the labels from the y_eval array.

Before training a Boosted Trees model, let's first train a linear classifier (logistic regression model). 
It is best practice to start with simpler model to establish a benchmark.
----------------------------------------------------------------------------------------------------------------
'''
linear_est = tf.estimator.LinearClassifier(feature_columns)

# Train model.
linear_est.train(train_input_fn, max_steps=100)

# Evaluation.
result = linear_est.evaluate(eval_input_fn)

print(pd.Series(result))

'''
-----------------------------------------------------------------------------------------------------------------
Next let's train a Boosted Trees model. 
For boosted trees, regression (BoostedTreesRegressor) and classification (BoostedTreesClassifier) are supported. 
Since the goal is to predict a class - survive or not survive, you will use the BoostedTreesClassifier.
-----------------------------------------------------------------------------------------------------------------
'''

# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset.
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

'''
------------------------------------------------------------------------------------------------------------------
Now you can use the train model to make predictions on a passenger from the evaluation set. 
TensorFlow models are optimized to make predictions on a batch, or collection, of examples at once. 
Earlier, the eval_input_fn is defined using the entire evaluation set.
------------------------------------------------------------------------------------------------------------------
'''

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()

'''
-------------------------------------------------------------------------------------------------------------------
Finally you can also look at the receiver operating characteristic (ROC) of the results, 
which will give us a better idea of the tradeoff between the true positive rate and false positive rate.
-------------------------------------------------------------------------------------------------------------------
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
        '       finished        estimator_boosted_tree.py             　                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()