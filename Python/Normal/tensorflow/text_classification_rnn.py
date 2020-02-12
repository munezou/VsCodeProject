'''
------------------------------------------------------------------------------------------
text
    Text classification with an RNN

This text classification tutorial trains a recurrent neural network 
on the IMDB large movie review dataset for sentiment analysis.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import io
import pprint
import contextlib
from pathlib import Path
from packaging import version
from PIL import Image
import tempfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

print(__doc__)


tfds.disable_progress_bar()

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

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup input pipeline                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The IMDB large movie review dataset is 
a binary classification dataset—all the reviews have either a positive or negative sentiment.

Download the dataset using TFDS.
---------------------------------------------------------------------------------------------------------------
'''
(train_data, test_data), info = tfds.load(
                                    name='imdb_reviews/subwords8k', 
                                    data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                                    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
                                    with_info=True, 
                                    as_supervised=True
                                )

'''
---------------------------------------------------------------------------------------------------------------
The dataset info includes the encoder (a tfds.features.text.SubwordTextEncoder).
---------------------------------------------------------------------------------------------------------------
'''
encoder = info.features['text'].encoder

print ('Vocabulary size: {}'.format(encoder.vocab_size))