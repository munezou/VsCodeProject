'''
------------------------------------------------------------------------------------------
tf.data
    text unicode
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
import tensorflow_datasets as tfds

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
        '       Reading text using tf.data                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This tutorial demonstrates how to load a sample from a text file using tf.data.TextLineDataset. 
TextLineDataset is designed for creating datasets from text files. 
Here, each line of the original text file is a sample. 
This may be useful for dealing with basically line-based text data (such as poetry or error logs).

This tutorial uses three different English translations of the same work, Homer's Iliad, 
to train a model to identify translators from a single line of text.
--------------------------------------------------------------------------------------------------------------
'''
'''
--------------------------------------------------------------------------------------------------------------
The text of the three translations is as follows:

* William Cowper — text

* Edward, Earl of Derby — text

* Samuel Butler — text

The text file used in this tutorial has gone through some typical preprocessing, 
such as removing headers, footers, line numbers, and chapter titles.
---------------------------------------------------------------------------------------------------------------
'''
# Let's download the file after preprocessing.
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

text_data_path = PROJECT_ROOT_DIR.joinpath('text_data')

for name in FILE_NAMES:
    train_file_path = tf.keras.utils.get_file(
            origin=DIRECTORY_URL+name,
            fname=text_data_path.joinpath(name)
        )

parent_dir = PROJECT_ROOT_DIR

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Read text into dataset.                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Iterate over the files and load each into a separate dataset.

Each sample requires labeling, so use tf.data.Dataset.map to apply the labeling function. 
This method iterates over all samples in the dataset and returns a (example, label) pair.
---------------------------------------------------------------------------------------------------------------
'''
def labeler(example, index):
    return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(text_data_path.joinpath(file_name)))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

# Combine the labeled datasets into a single dataset and shuffle.
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]

for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE, 
        reshuffle_each_iteration=False
    )

print()

'''
---------------------------------------------------------------------------------------------------------------
You can use tf.data.Dataset.take and print to see what the (example, label) pair looks like.
The numpy property indicates the value of each tensor.
---------------------------------------------------------------------------------------------------------------
'''
print('---< output (example, label) >---')
for ex in all_labeled_data.take(5):
    print(ex)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Encode text lines into numbers.                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Strings need to be converted to lists of numbers, because machine learning models deal with numbers, not words.
Therefore, map unique words to unique numbers.
---------------------------------------------------------------------------------------------------------------
'''
'''
---------------------------------------------------------------------------------------------------------------
Building a vocabulary

First, tokenize the text and build a vocabulary as a collection of individual unique words. 
There are several ways to do this using TensorFlow or Python. Here we do the following:

1. Iterate through the numpy values for each sample.
2. Use tfds.features.text.Tokenizer to split it into tokens.
3. Aggregate the tokens into a Python set to eliminate duplication.
4. Get the vocabulary size for later use.
----------------------------------------------------------------------------------------------------------------
'''
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()

for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print('vocab_size = {0}\n'.format(vocab_size))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Encode the sample.                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Pass the vocabulary_set to tfds.features.text.TokenTextEncoder to create an encoder.
# The encoder's encode method takes a text string and returns a list of integers.
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

'''
----------------------------------------------------------------------------------------------------------------
You can apply this to just one line and see what the output looks like.
----------------------------------------------------------------------------------------------------------------
'''
example_text = next(iter(all_labeled_data))[0].numpy()
print('example_text = \n{0}\n'.format(example_text))

encoded_example = encoder.encode(example_text)
print('encoded_example = \n{0}\n'.format(encoded_example))

# Then wrap the encoder in tf.py_function and pass it to the dataset's map method to apply it to the dataset.
def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

all_encoded_data = all_labeled_data.map(encode_map_fn)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Divide the data set into test and training batches.                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use tf.data.Dataset.take and tf.data.Dataset.skip to create a small test dataset and a larger training set.

Before passing to the model, the dataset must be batched.
Usually, the samples in a batch must be the same size and shape.
However, the samples in these datasets are not all the same size.
Each line of text has a different number of words.
For this reason, pad the sample to the same size using the tf.data.Dataset.padded_batch method (instead of batch).
---------------------------------------------------------------------------------------------------------------
'''
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([-1],[]))

'''
---------------------------------------------------------------------------------------------------------------
Now, test_data and train_data are not collections of (example, label) pairs, but collections of batches.
Each batch is an array pair (many samples, many labels).
---------------------------------------------------------------------------------------------------------------
'''
sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]

'''
----------------------------------------------------------------------------------------------------------------
The introduction of one new token number (using zeros for padding) has increased the vocabulary size by one.
----------------------------------------------------------------------------------------------------------------
'''
vocab_size += 1

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Build the model.                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
model = tf.keras.Sequential()

'''
---------------------------------------------------------------------------------------------------------------
The first layer converts the integer representation to dense vector embedding.
See the Word Embedding Tutorial for details.
---------------------------------------------------------------------------------------------------------------
'''
model.add(tf.keras.layers.Embedding(vocab_size, 64))

'''
---------------------------------------------------------------------------------------------------------------
The next layer is the Long Short-Term Memory layer.
With this layer, the model interprets words in the context of other words. 
LSTM's Bidirectional wrapper allows you to learn data points in relation to data points before and after.
---------------------------------------------------------------------------------------------------------------
'''
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

'''
---------------------------------------------------------------------------------------------------------------
Finally, there is one or more fully connected layers, and the last layer is the output layer.
The output layer generates probabilities for all labels.
More complex but probable labels are the sample labels that the model predicts.
--------------------------------------------------------------------------------------------------------------
'''
# One or more dense layers
# Edit the list in the `for` line and experiment with layer sizes
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer First argument is the number of labels
model.add(tf.keras.layers.Dense(3, activation='softmax'))

'''
--------------------------------------------------------------------------------------------------------------
Finally, compile the model.
Softmax categorization models use sparse_categorical_crossentropy as the loss function.
You can use other optimizers, but adam is often used.
-------------------------------------------------------------------------------------------------------------
'''
model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model.                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Applying this model to this data gives decent results (about 83%).
---------------------------------------------------------------------------------------------------------------
'''
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {0}, Eval accuracy: {1}'.format(eval_loss, eval_acc))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Unicode strings                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Introduction

Models that process natural language often handle different languages with different character sets. 
Unicode is a standard encoding system that is used to represent character from almost all languages. 
Each character is encoded using a unique integer code point between 0 and 0x10FFFF.
A Unicode string is a sequence of zero or more code points.

This tutorial shows how to represent Unicode strings in TensorFlow and manipulate them using Unicode equivalents of standard string ops. 
It separates Unicode strings into tokens based on script detection.
--------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The tf.string data type                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The basic TensorFlow tf.string dtype allows you to build tensors of byte strings. 
Unicode strings are utf-8 encoded by default.
---------------------------------------------------------------------------------------------------------------
'''
tf_constant_thanks = tf.constant(u"Thanks 😊")
print('tf_constant_thanks = \n{0}\n'.format(tf_constant_thanks))