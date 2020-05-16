'''
------------------------------------------------------------------------------------------
estimator
    Create an Estimator from a Keras model

Overview
TensorFlow Estimators are fully supported in TensorFlow, and can be created from new and existing tf.keras models. 
This tutorial contains a complete, minimal example of that process.
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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create a simple Keras model.                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
In Keras, you assemble layers to build models. 
A model is (usually) a graph of layers. 
The most common type of model is a stack of layers: the tf.keras.Sequential model.

To build a simple, fully-connected network (i.e. multi-layer perceptron):
--------------------------------------------------------------------------------------------------------------
'''

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# Compile the model and get a summary.
model.compile(loss='categorical_crossentropy', optimizer='adam')
print()

print('model.summary() = \n{0}\n'.format(model.summary()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create an input function                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Use the Datasets API to scale to large datasets or multi-device training.

Estimators need control of when and how their input pipeline is built. 
To allow this, they require an "Input function" or input_fn. 
The Estimator will call this function with no arguments. 
The input_fn must return a tf.data.Dataset.
---------------------------------------------------------------------------------------------------------------
'''
def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset

# Test out your input_fn
for features_batch, labels_batch in input_fn().take(1):
    print(features_batch)
    print(labels_batch)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create an Estimator from the tf.keras model.                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
A tf.keras.Model can be trained with the tf.estimator API 
by converting the model to an tf.estimator.Estimator object with tf.keras.estimator.model_to_estimator.
----------------------------------------------------------------------------------------------------------------
'''
model_dir = PROJECT_ROOT_DIR.joinpath('tmp', 'tfkeras_example')

keras_estimator = tf.keras.estimator.model_to_estimator(
                    keras_model=model, 
                    model_dir=model_dir
                )

# Train and evaluate the estimator.
keras_estimator.train(input_fn=input_fn, steps=25)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        estimator_from_keras_model.py             　        (2020/05/16)               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()