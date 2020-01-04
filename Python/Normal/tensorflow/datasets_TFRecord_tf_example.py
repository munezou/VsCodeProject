'''
------------------------------------------------------------------------------------------
tf.data
    Using TFRecords and tf.Example

For efficient reading of data, it is useful to serialize the data and store 
it in a set of files that can be read continuously (each file is 100-200MB).
This is especially true when trying to stream data over a network.
It is also useful for caching data preprocessing.

The TFRecord format is a simple format for storing a series of binary records.

Protocol buffers are platform and language independent libraries that efficiently serialize structured data.

Protocol messages are represented by files with a .proto extension. 
The easiest way to identify the message type.

The tf.Example message (or protocol buffer) is a flexible message type that represents a mapping of the form "" string ": value}.
It is designed for TensorFlow and is commonly used by higher-level APIs like TFX.

This notebook demonstrates how to create, parse, and use a tf.Example message, 
then shows how to serialize the tf.Example message, write it to a .tfrecord, and then read it.

note:
While these structures are useful, they do not have to be.
Unless you are using tf.data and reading data is still a training bottleneck, 
you do not need to change your existing code to use TFRecords.
See Data Input Pipeline Performance for tips on improving dataset performance.
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
        '       tf.Example                                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Data type for tf.Example
Basically, tf.Example is a mapping of {"string": tf.train.Feature}.

 The tf.train.Feature message type can take one of three types: (See .proto file.) Most other common data types can be forced to one of these.

1. tf.train.BytesList (can handle the following types of data)

* string
* byte

2. tf.train.FloatList (can handle the following types of data)

* float (float32)
* double (float64)

3. tf.train.Int64List (can handle the following types of data)

* bool
* enum
* int32
* uint32
* int64
* uint64

To convert a normal TensorFlow type to a tf.train.Feature compatible with tf.Example, you can use the following shortcut function:

Each function takes a single scalar value and returns a tf.train.Feature containing one of the three list types above.
------------------------------------------------------------------------------------------------------------------
'''

# The function below can be used to convert a value to a type compatible with tf.Example.

def _bytes_feature(value):
    """
    ---------------------------------------------------------
    Returns byte_list from string / byte type.
    ---------------------------------------------------------
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """
    ---------------------------------------------------------
    Returns float_list from float / double type.
    ---------------------------------------------------------
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """
    ---------------------------------------------------------
    Returns Int64_list from bool / enum / int / uint type.
    ---------------------------------------------------------
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
-----------------------------------------------------------------------------------------
note:
For simplicity, this sample only deals with scalar value input.
The easiest way to handle non-scalar features is to convert the tensor to a binary string using tf.serialize_tensor. 
TensorFlow treats strings as scalar values.
To convert a binary string back into a tensor, use tf.parse_tensor.

An example of using the above function is shown below.
Notice that the inputs are of different types, while the outputs are standardized.
An exception is raised if the input is of a type that cannot be coerced. 
(Example: _int64_feature (1.0) is an error, because 1.0 is a floating point number, you should use the _float_feature function instead)
------------------------------------------------------------------------------------------
'''
print()
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

'''
-------------------------------------------------------------------------------------------
All major messages can be serialized to binary strings using .SerializeToString.
-------------------------------------------------------------------------------------------
'''

feature = _float_feature(np.exp(1))
print('feature.SerializeToString() = \n{0}\n'.format(feature.SerializeToString()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Creating a tf.Example message                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Suppose you want to create a tf.Example from existing data.
Actually, the source of the data set can be anywhere, but the procedure for creating a tf.Example message from one observation record is the same.

1. For each observation, each value must be converted to a tf.train.Feature of three compatible types using the above function.

2. Next, create a map (dictionary) in which the character strings representing the feature names correspond to the encoded features created in # 1.

3. Convert the map created in step 2 into a feature message.
In this notebook, you create a dataset using NumPy.

This data set has four features.

* Logical value representing False or True. The appearance probabilities are assumed to be equal.
* Random byte value. Suppose it is uniform throughout.
* An integer value uniformly sampled from the range [-10000, 10000).
* Floating point numbers sampled from the standard normal distribution.

The sample shall consist of 10,000 observation records distributed in a similar manner independently of the above distribution.
----------------------------------------------------------------------------------------------------------------
'''
# Number of observations included in the dataset
n_observations = int(1e4)

# Boolean feature encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature value-random number between -10000 and 10000
feature1 = np.random.randint(0, 5, n_observations)

# Byte feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Floating point features Generated from the standard normal distribution.
feature3 = np.random.randn(n_observations)

'''
----------------------------------------------------------------------------------------------------------------
These features are coerced to a tf.Example compatible type using one of _bytes_feature, _float_feature, _int64_feature.
You can then create a tf.Example message from the encoded features.
----------------------------------------------------------------------------------------------------------------
'''
def serialize_example(feature0, feature1, feature2, feature3):
    """
    --------------------------------------------------------
    Create a tf.Example message that can be output to a file.
    --------------------------------------------------------
    """
    
    # Create a dictionary that maps feature names to tf.Example compatible data types.
    
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    
    # Create a feature message using tf.train.Example.
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

'''
-------------------------------------------------------------------------------------------------------------------
For example, suppose your dataset has one observation, [False, 4, bytes ('goat'), 0.9876].
You can create and print a tf.Example message from this observation using create_message ().
As mentioned above, each observation record is written as a Features message.
Note that the tf.Example message is just a wrapper around this Features message.
--------------------------------------------------------------------------------------------------------------------
'''
# データセットからの観測記録の例

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)


