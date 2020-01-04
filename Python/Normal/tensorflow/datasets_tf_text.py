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
import tensorflow_text as text

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

