'''
------------------------------------------------------------------------------------------
text
    Transformer model for language understanding

This tutorial trains a Transformer model to translate Portuguese to English. 
This is an advanced example that assumes knowledge of text generation and attention.

The core idea behind the Transformer model is self-attention—the ability to 
attend to different positions of the input sequence to compute a representation of that sequence. 
Transformer creates stacks of self-attention layers 
and is explained below in the sections Scaled dot product attention and Multi-head attention.

A transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. 
This general architecture has a number of advantages:

	* It make no assumptions about the temporal/spatial relationships across the data. 
		This is ideal for processing a set of objects (for example, StarCraft units).
	* Layer outputs can be calculated in parallel, instead of a series like an RNN.
	* Distant items can affect each other's output without passing through many RNN-steps, 
		or convolution layers (see Scene Memory Transformer for example).
	* It can learn long-range dependencies. This is a challenge in many sequence tasks.
	
The downsides of this architecture are:

	* For a time-series, 
		the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. 
		This may be less efficient.
	* If the input does have a temporal/spatial relationship, like text, 
		some positional encoding must be added or the model will effectively see a bag of words.
	
After training the model in this notebook, you will be able to input a Portuguese sentence and return the English translation.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import json
import time
import pprint
import contextlib
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow_datasets as tfds

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/attention_map_portuguese.png'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Setup input pipeline                                                       \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Use TFDS to load the Portugese-English translation dataset from the TED Talks Open Translation Project.

This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.
------------------------------------------------------------------------------------------
'''
examples, metadata = tfds.load(
                        name='ted_hrlr_translate/pt_to_en',
                        data_dir=str(PROJECT_ROOT_DIR.joinpath('Data')),
                        with_info=True,
                        as_supervised=True
                    )

train_examples, val_examples = examples['train'], examples['validation']

'''
-----------------------------------------------------------------------------------------
Create a custom subwords tokenizer from the training dataset.
-----------------------------------------------------------------------------------------
'''

