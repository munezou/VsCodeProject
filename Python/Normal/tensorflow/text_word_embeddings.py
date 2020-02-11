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


from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

print(__doc__)

from tensorflow_examples.models.pix2pix import pix2pix

keras = tf.keras
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
