'''
------------------------------------------------------------------------------------------
text
    Word embeddings

This tutorial introduces word embeddings. 
It contains complete code to train word embeddings from scratch on a small dataset, 
and to visualize these embeddings using the Embedding Projector (shown in the image below).

Representing text as numbers
    Machine learning models take vectors (arrays of numbers) as input. 
    When working with text, the first thing we must do come up with a strategy to convert strings to numbers 
    (or to "vectorize" the text) before feeding it to the model. 
    In this section, we will look at three strategies for doing so.

------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
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

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/embedding.jpg'))
im.show()

'''
---------------------------------------------------------------------------------------------------------------
Representing text as numbers
    Machine learning models take vectors (arrays of numbers) as input. 
    When working with text, the first thing we must do come up with a strategy to convert strings to numbers 
    (or to "vectorize" the text) before feeding it to the model. 
    In this section, we will look at three strategies for doing so.
---------------------------------------------------------------------------------------------------------------
'''
'''
---------------------------------------------------------------------------------------------------------------
One-hot encodings
    As a first idea, we might "one-hot" encode each word in our vocabulary. 
    Consider the sentence "The cat sat on the mat". 
    The vocabulary (or unique words) in this sentence is (cat, mat, on, sat, the). 
    To represent each word, we will create a zero vector with length equal to the vocabulary, 
    then place a one in the index that corresponds to the word. 
    This approach is shown in the following diagram.
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/one-hot.png'))
im.show()

'''
----------------------------------------------------------------------------------------------------------------
To create a vector that contains the encoding of the sentence, 
we could then concatenate the one-hot vectors for each word.

Key point: 
This approach is inefficient. A one-hot encoded vector is sparse (meaning, most indicices are zero). 
Imagine we have 10,000 words in the vocabulary. 
To one-hot encode each word, we would create a vector where 99.99% of the elements are zero.
----------------------------------------------------------------------------------------------------------------
'''

'''
---------------------------------------------------------------------------------------------------------------
Encode each word with a unique number

A second approach we might try is to encode each word using a unique number. 
Continuing the example above, we could assign 1 to "cat", 2 to "mat", and so on. 
We could then encode the sentence "The cat sat on the mat" as a dense vector like [5, 1, 4, 3, 5, 2]. 
This appoach is efficient. Instead of a sparse vector, 
we now have a dense one (where all elements are full).

There are two downsides to this approach, however:

    * The integer-encoding is arbitrary (it does not capture any relationship between words).

    * An integer-encoding can be challenging for a model to interpret. 
    A linear classifier, for example, learns a single weight for each feature. 
    Because there is no relationship between the similarity of any two words and the similarity of their encodings, 
    this feature-weight combination is not meaningful.
--------------------------------------------------------------------------------------------------------------
'''
'''
--------------------------------------------------------------------------------------------------------------
Word embeddings

Word embeddings give us a way to use an efficient, 
dense representation in which similar words have a similar encoding. Importantly, 
we do not have to specify this encoding by hand. 
An embedding is a dense vector of floating point values (the length of the vector is a parameter you specify). 
Instead of specifying the values for the embedding manually, 
they are trainable parameters (weights learned by the model during training, in the same way a model learns weights for a dense layer). 
It is common to see word embeddings that are 8-dimensional (for small datasets), up to 1024-dimensions when working with large datasets. 
A higher dimensional embedding can capture fine-grained relationships between words, but takes more data to learn.
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/embedding2.png'))
im.show()

'''
---------------------------------------------------------------------------------------------------------------
Above is a diagram for a word embedding. 
Each word is represented as a 4-dimensional vector of floating point values. 
Another way to think of an embedding is as "lookup table". 
After these weights have been learned, 
we can encode each word by looking up the dense vector it corresponds to in the table.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Using the Embedding layer                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Keras makes it easy to use word embeddings. 
Let's take a look at the Embedding layer.

The Embedding layer can be understood as a lookup table that maps 
from integer indices (which stand for specific words) to dense vectors (their embeddings). 
The dimensionality (or width) of the embedding is a parameter you can experiment with to see what works well for your problem, 
much in the same way you would experiment with the number of neurons in a Dense layer.
-------------------------------------------------------------------------------------------------------------
'''
embedding_layer = layers.Embedding(1000, 5)

'''
------------------------------------------------------------------------------------------------------------
When you create an Embedding layer, 
the weights for the embedding are randomly initialized (just like any other layer). 
During training, they are gradually adjusted via backpropagation. 
Once trained, the learned word embeddings will roughly encode similarities between words 
(as they were learned for the specific problem your model is trained on).

If you pass an integer to an embedding layer, 
the result replaces each integer with the vector from the embedding table:
-------------------------------------------------------------------------------------------------------------
'''
result = embedding_layer(tf.constant([1,2,3]))

print('result.numpy() = \n{0}\n'.format(result.numpy()))

'''
-------------------------------------------------------------------------------------------------------------
For text or sequence problems, 
the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length), 
where each entry is a sequence of integers. 
It can embed sequences of variable lengths. 
You could feed into the embedding layer 
above batches with shapes (32, 10) (batch of 32 sequences of length 10) or (64, 15) (batch of 64 sequences of length 15).

The returned tensor has one more axis than the input, the embedding vectors are aligned along the new last axis. 
Pass it a (2, 3) input batch and the output is (2, 3, N)
----------------------------------------------------------------------------------------------------------------
'''
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))

print('result.shape = \n{0}\n'.format(result.shape))

'''
----------------------------------------------------------------------------------------------------------------
When given a batch of sequences as input, an embedding layer returns a 3D floating point tensor, 
of shape (samples, sequence_length, embedding_dimensionality). 
To convert from this sequence of variable length to a fixed representation there are a variety of standard approaches. 
You could use an RNN, Attention, or pooling layer before passing it to a Dense layer. 
This tutorial uses pooling because it's simplest. 
The Text Classification with an RNN tutorial is a good next step.
----------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Learning embeddings from scratch                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
In this tutorial you will train a sentiment classifier on IMDB movie reviews. 
In the process, the model will learn embeddings from scratch. 
We will use to a preprocessed dataset.

To load a text dataset from scratch see the Loading text tutorial.
----------------------------------------------------------------------------------------------------------------
'''
(train_data, test_data), info = tfds.load(
                                    name='imdb_reviews/subwords8k', 
                                    data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                                    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
                                    with_info=True, 
                                    as_supervised=True
                                )

'''
----------------------------------------------------------------------------------------------------------------
Get the encoder (tfds.features.text.SubwordTextEncoder), 
and have a quick look at the vocabulary.

The "_" in the vocabulary represent spaces. 
Note how the vocabulary includes whole words (ending with "_") 
and partial words which it can use to build larger words:
----------------------------------------------------------------------------------------------------------------
'''
encoder = info.features['text'].encoder

print('encoder.subwords[:20] = \n{0}\n'.format(encoder.subwords[:20]))

'''
----------------------------------------------------------------------------------------------------------------
Movie reviews can be different lengths. 
We will use the padded_batch method to standardize the lengths of the reviews.
----------------------------------------------------------------------------------------------------------------
'''
padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)

