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