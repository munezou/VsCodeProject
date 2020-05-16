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

import tensorflow_datasets as tfds

print(__doc__)


tfds.disable_progress_bar()

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
name_space = os.path.join('imdb_reviews', 'subwords8k')


(train_dataset, test_dataset), info = tfds.load(
                                    name=name_space, 
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

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
    print ('{} ----> {}'.format(index, encoder.decode([index])))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Prepare the data for training                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Next create batches of these encoded strings. 
Use the padded_batch method to zero-pad the sequences to the length of the longest string in the batch:
---------------------------------------------------------------------------------------------------------------
'''
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Build a tf.keras.Sequential model and start with an embedding layer. 
An embedding layer stores one vector per word. 
When called, it converts the sequences of word indices to sequences of vectors. 
These vectors are trainable. 
After training (on enough data), words with similar meanings often have similar vectors.

This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.

A recurrent neural network (RNN) processes sequence input by iterating through the elements. 
RNNs pass the outputs from one timestep to their input—and then to the next.

The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer. 
This propagates the input forward and backwards through the RNN layer and then concatenates the output. 
This helps the RNN to learn long range dependencies.
----------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

'''
----------------------------------------------------------------------------------------------------------------
Compile the Keras model to configure the training process:
----------------------------------------------------------------------------------------------------------------
'''
model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
history = model.fit(
                train_dataset, epochs=10,
                validation_data=test_dataset, 
                validation_steps=30
            )

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

'''
-----------------------------------------------------------------------------------------------------------------
The above model does not mask the padding applied to the sequences. 
This can lead to skew if trained on padded sequences and test on un-padded sequences. 
Ideally you would use masking to avoid this, but as you can see below it only have a small effect on the output.

If the prediction is >= 0.5, it is positive else it is negative.
-----------------------------------------------------------------------------------------------------------------
'''
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)

# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Stack two or more LSTM layers                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Keras recurrent layers have two available modes that are controlled by the return_sequences constructor argument:

* Return either the full sequences of successive outputs for each timestep (a 3D tensor of shape (batch_size, timesteps, output_features)).

* Return only the last output for each input sequence (a 2D tensor of shape (batch_size, output_features)).
---------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy']
            )

history = model.fit(
                train_dataset, 
                epochs=10,
                validation_data=test_dataset,
                validation_steps=30
            )

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# predict on a sample text without padding.

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

'''
------------------------------------------------------------------------------------
Check out other existing recurrent layers such as GRU layers.

If you're interestied in building custom RNNs, see the Keras RNN Guide.
------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        text_classification_rnn.py                   　                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()