'''
------------------------------------------------------------------------------------------
distribute
    Distributed training with Keras

Overview
The tf.distribute.Strategy API provides an abstraction for distributing your training across multiple processing units. 
The goal is to allow users to enable distributed training using existing models and training code, with minimal changes.

This tutorial uses the tf.distribute.MirroredStrategy, which does in-graph replication with synchronous training on many GPUs on one machine. 
Essentially, it copies all of the model's variables to each processor. 
Then, it uses all-reduce to combine the gradients from all processors and applies the combined value to all copies of the model.

MirroredStrategy is one of several distribution strategy available in TensorFlow core. 
You can read about more strategies at distribution strategy guide.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import contextlib
from pathlib import Path
from packaging import version
from PIL import Image
import tempfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Keras API                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This example uses the tf.keras API to build the model and training loop. 
For custom training loops, see the tf.distribute.Strategy with training loops tutorial.
---------------------------------------------------------------------------------------------------------------
'''
