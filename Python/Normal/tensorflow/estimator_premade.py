'''
------------------------------------------------------------------------------------------
estimator
    Premade Estimators

This tutorial shows you how to solve the Iris classification problem in TensorFlow using Estimators. 
An Estimator is TensorFlow's high-level representation of a complete model, 
and it has been designed for easy scaling and asynchronous training. 
For more details see Estimators.

Note that in TensorFlow 2.0, the Keras API can accomplish many of these same tasks, 
and is believed to be an easier API to learn. 
If you are starting fresh, we would recommend you start with Keras. 
For more information about the available high level APIs in TensorFlow 2.0, 
see Standardizing on Keras.
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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The data set                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The sample program in this document builds and tests a model that classifies Iris flowers into three different species 
based on the size of their sepals and petals.

You will train a model using the Iris data set. 
The Iris data set contains four features and one label.
The four features identify the following botanical characteristics of individual Iris flowers:

* sepal length
* sepal width
* petal length
* petal width

Based on this information, you can define a few helpful constants for parsing the data:
-----------------------------------------------------------------------------------------------------------------
'''
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Next, download and parse the Iris data set using Keras and Pandas. 
# Note that you keep distinct datasets for training and testing.
iris_folder_path = PROJECT_ROOT_DIR.joinpath('csv_data', 'iris')
train_path = tf.keras.utils.get_file(
        str(iris_folder_path.joinpath('iris_training.csv')), 
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    )

test_path = tf.keras.utils.get_file(
        str(iris_folder_path.joinpath('iris_test.csv')),    
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    )

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

'''
-----------------------------------------------------------------------------------------------------------------
You can inspect your data to see that you have four float feature columns and one int32 label.
-----------------------------------------------------------------------------------------------------------------
'''
print('train.head() = \n{0}\n'.format(train.head()))

'''
-----------------------------------------------------------------------------------------------------------------
For each of the datasets, split out the labels, which the model will be trained to predict.
-----------------------------------------------------------------------------------------------------------------
'''
train_y = train.pop('Species')
test_y = test.pop('Species')

# The label column has now been removed from the features.
print('train.head() = \n{0}\n'.format(train.head()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Overview of programming with Estimators                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To write a TensorFlow program based on pre-made Estimators, you must perform the following tasks:

* Create one or more input functions.
* Define the model's feature columns.
* Instantiate an Estimator, specifying the feature columns and various hyperparameters.
* Call one or more methods on the Estimator object, passing the appropriate input function as the source of the data.

Let's see how those tasks are implemented for Iris classification.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create input functions                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
----------------------------------------------------------------------------------------------------------------
You must create input functions to supply data for training, evaluating, and prediction.

An input function is a function that returns a tf.data.Dataset object which outputs the following two-element tuple:

*features - A Python dictionary in which:
    * Each key is the name of a feature.
    * Each value is an array containing all of that feature's values.
*label - An array containing the values of the label for every example.
Just to demonstrate the format of the input function, here's a simple implementation:
----------------------------------------------------------------------------------------------------------------
'''
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels

'''
----------------------------------------------------------------------------------------------------------------
Your input function may generate the features dictionary and label list any way you like. 
However, we recommend using TensorFlow's Dataset API, which can parse all sorts of data.

The Dataset API can handle a lot of common cases for you. 
For example, using the Dataset API, 
you can easily read in records from a large collection of files in parallel and join them into a single stream.

To keep things simple in this example you are going to load the data with pandas, 
and build an input pipeline from this in-memory data:
----------------------------------------------------------------------------------------------------------------
'''
def input_fn(features, labels, training=True, batch_size=256):
    """
    -------------------------------------------------
    An input function for training or evaluating
    -------------------------------------------------
    """
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the feature columns                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A feature column is an object describing how the model should use raw input data from the features dictionary. 
When you build an Estimator model, 
you pass it a list of feature columns that describes each of the features you want the model to use. 
The tf.feature_column module provides many options for representing data to the model.

For Iris, the 4 raw features are numeric values, 
so we'll build a list of feature columns to tell the Estimator model to represent each of the four features as 32-bit floating-point values. 
Therefore, the code to create the feature column is:
---------------------------------------------------------------------------------------------------------------
'''
# Feature columns describe how to use the input.
my_feature_columns = []

for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

'''
----------------------------------------------------------------------------------------------------------------
Feature columns can be far more sophisticated than those we're showing here. 
You can read more about Feature Columns in this guide.

Now that you have the description of how you want the model to represent the raw features, you can build the estimator.
----------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Instantiate an estimator                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
----------------------------------------------------------------------------------------------------------------
The Iris problem is a classic classification problem. 
Fortunately, TensorFlow provides several pre-made classifier Estimators, including:

* tf.estimator.DNNClassifier for deep models that perform multi-class classification.
* tf.estimator.DNNLinearCombinedClassifier for wide & deep models.
* tf.estimator.LinearClassifier for classifiers based on linear models.

For the Iris problem, tf.estimator.DNNClassifier seems like the best choice. 
Here's how you instantiated this Estimator:
---------------------------------------------------------------------------------------------------------------
'''
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
                feature_columns=my_feature_columns,
                # Two hidden layers of 30 and 10 nodes respectively.
                hidden_units=[30, 10],
                # The model must choose between 3 classes.
                n_classes=3
            )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train, Evaluate, and Predict                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Now that you have an Estimator object, you can call methods to do the following:

* Train the model.
* Evaluate the trained model.
* Use the trained model to make predictions.
-------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------------------------
Train the model by calling the Estimator's train method as follows:
------------------------------------------------------------------------------------------------------------
'''
# Train the Model.
classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True),
        steps=5000
    )

'''
----------------------------------------------------------------------------------------------------------
Note that you wrap up your input_fn call in a lambda to capture the arguments 
while providing an input function that takes no arguments, as expected by the Estimator. 
The steps argument tells the method to stop training after a number of training steps.
---------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Evaluate the trained model                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------
Now that the model has been trained, 
you can get some statistics on its performance. 
The following code block evaluates the accuracy of the trained model on the test data:
----------------------------------------------------------------------------------------------------------
'''
eval_result = classifier.evaluate(
                    input_fn=lambda: input_fn(test, test_y, training=False)
                )

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

'''
------------------------------------------------------------------------------------------------------------
Unlike the call to the train method, you did not pass the steps argument to evaluate. 
The input_fn for eval only yields a single epoch of data.

The eval_result dictionary also contains the average_loss (mean loss per sample), 
the loss (mean loss per mini-batch) and the value of the estimator's global_step 
(the number of training iterations it underwent).
-------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Making predictions (inferring) from the trained model                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
-------------------------------------------------------------------------------------------------------------
You now have a trained model that produces good evaluation results. 
You can now use the trained model to predict the species of an Iris flower based on some unlabeled measurements. 
As with training and evaluation, you make predictions using a single function call:
-------------------------------------------------------------------------------------------------------------
'''
# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

'''
------------------------------------------------------------------------------------------------------------
The predict method returns a Python iterable, yielding a dictionary of prediction results for each example. 
The following code prints a few predictions and their probabilities:
-----------------------------------------------------------------------------------------------------------
'''

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        estimator_premade.py     　　　　　       　                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()