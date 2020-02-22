'''
------------------------------------------------------------------------------------------
text
    Neural machine translation with attention

This notebook trains a sequence to sequence (seq2seq) model for Spanish to English translation. 
This is an advanced example that assumes some knowledge of sequence to sequence models.

After training the model in this notebook, you will be able to input a Spanish sentence, 
such as *"¿todavia estan en casa?", and return the English translation: *"are you still at home?"

The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting. 
This shows which parts of the input sentence has the model's attention while translating:
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import unicodedata
import re
import os
import io
import time
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
import matplotlib.ticker as ticker

import tensorflow as tf

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

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/spanish-english.png'))
im.show()
'''
---------------------------------------------------------------------------------------------------------------
Note: This example takes approximately 10 mintues to run on a single P100 GPU.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      Download and prepare the dataset                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
We'll use a language dataset provided by http://www.manythings.org/anki/. 
This dataset contains language translation pairs in the format:

May I borrow this book?    ¿Puedo tomar prestado este libro?
There are a variety of languages available, but we'll use the English-Spanish dataset. 
For convenience, we've hosted a copy of this dataset on Google Cloud, but you can also download your own copy. 
After downloading the dataset, here are the steps we'll take to prepare the data:

	1. Add a start and end token to each sentence.
	2. Clean the sentences by removing special characters.
	3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
	4. Pad each sentence to a maximum length.
--------------------------------------------------------------------------------------------------------------
'''
spa_eng_path = str(PROJECT_ROOT_DIR.joinpath('Data/spa_eng/spa-eng.zip'))

# Download the file
path_to_zip = tf.keras.utils.get_file(
                fname=spa_eng_path, 
                origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                extract=True,
                cache_dir=PROJECT_ROOT_DIR.joinpath('Data/spa_eng')
            )

path_to_file = str(PROJECT_ROOT_DIR.joinpath('Data/spa_eng/datasets/spa-eng/spa.txt'))

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

