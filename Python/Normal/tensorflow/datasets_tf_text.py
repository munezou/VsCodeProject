'''
------------------------------------------------------------------------------------------
tf.data
    TF.Text

Introduction

TensorFlow Text provides a collection of text related classes and ops ready to use with TensorFlow 2.0. 
The library can perform the preprocessing regularly required by text-based models, 
and includes other features useful for sequence modeling not provided by core TensorFlow.

The benefit of using these ops in your text preprocessing is that they are done in the TensorFlow graph. 
You do not need to worry about tokenization in training being different than the tokenization at inference, 
or managing preprocessing scripts.
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
        '       Unicode                                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Most ops expect that the strings are in UTF-8. 
If you're using a different encoding, you can use the core tensorflow transcode op to transcode into UTF-8. 
You can also use the same op to coerce your string to structurally valid UTF-8 if your input could be invalid.
---------------------------------------------------------------------------------------------------------------
'''
docs = tf.constant(['Everything not saved will be lost.'.encode('UTF-16-BE'), 'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Tokenization                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Tokenization is the process of breaking up a string into tokens. 
Commonly, these tokens are words, numbers, and/or punctuation.

The main interfaces are Tokenizer and TokenizerWithOffsets which each have a single method tokenize and tokenize_with_offsets respectively. 
There are multiple tokenizers available now. 
Each of these implement TokenizerWithOffsets (which extends Tokenizer) which includes an option for getting byte offsets into the original string. 
This allows the caller to know the bytes in the original string the token was created from.

All of the tokenizers return RaggedTensors with the inner-most dimension of tokens mapping to the original individual strings. 
As a result, the resulting shape's rank is increased by one. 
Please review the ragged tensor guide if you are unfamiliar with them. https://www.tensorflow.org/guide/ragged_tensors
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        datasets_tf_text.py                          (2020/05/16)                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()