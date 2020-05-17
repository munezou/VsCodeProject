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
import datetime

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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
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
# Example of observation record from dataset

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)

print('serialized_example = \n{0}\n'.format(serialized_example))

# Use the tf.train.Example.FromString method to decode the message.
example_proto = tf.train.Example.FromString(serialized_example)

print('example_proto = \n{0}\n'.format(example_proto))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       TFRecord format details                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A TFRecord file contains a sequence of records. The file can only be read sequentially.

Each record contains a byte-string, for the data-payload, plus the data-length, 
and CRC32C (32-bit CRC using the Castagnoli polynomial) hashes for integrity checking.


The format of each record is:

Each record is stored in the following formats:
uint64 length
uint32 masked_crc32_of_length
byte   data[length]
uint32 masked_crc32_of_data

The records are concatenated together to produce the file. CRCs are described here, and the mask of a CRC is:
masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul

Note: 
There is no requirement to use tf.Example in TFRecord files. 
tf.Example is just a method of serializing dictionaries to byte-strings. 
Lines of text, encoded image data, or serialized tensors (using tf.io.serialize_tensor, and tf.io.parse_tensor when loading). 
See the tf.io module for more options.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       TFRecord files using tf.data                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The tf.data module also provides tools for reading and writing data in TensorFlow.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Writing a TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
The easiest way to get the data into a dataset is to use the from_tensor_slices method.
--------------------------------------------------------------------------------------------------------------
'''
# Applied to an array, it returns a dataset of scalars:
tf_data_Dataset_from_tensor_slices = tf.data.Dataset.from_tensor_slices(feature1)

print('tf_data_Dataset_from_tensor_slices = \n{0}\n'.format(tf_data_Dataset_from_tensor_slices))

# Applies to a tuple of arrays, it returns a dataset of tuples:
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))

print('features_dataset = \n{0}\n'.format(features_dataset))

# Use `take(1)` to only pull one example from the dataset.
for f0,f1,f2,f3 in features_dataset.take(1):
    print('f0 = {0}\n'.format(f0))
    print('f1 = {0}\n'.format(f1))
    print('f2 = {0}\n'.format(f2))
    print('f3 = {0}\n'.format(f3))

'''
--------------------------------------------------------------------------------------------------------------
Use the tf.data.Dataset.map method to apply a function to each element of a Dataset.

The mapped function must operate in TensorFlow graph mode—it must operate on and return tf.Tensors. 
A non-tensor function, like serialize_example, can be wrapped with tf.py_function to make it compatible.

Using tf.py_function requires to specify the shape and type information that is otherwise unavailable:
-------------------------------------------------------------------------------------------------------------
'''


def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
                    func=serialize_example,
                    inp=(f0,f1,f2,f3),
                    Tout=tf.string
                )
    
    return tf.reshape(tf_string, ()) # The result is a scalar


tf_serialize_example_print = tf_serialize_example(f0,f1,f2,f3)
print('tf_serialize_example_print = \n{0}\n'.format(tf_serialize_example_print))

# Apply this function to each element in the dataset:
serialized_features_dataset = features_dataset.map(tf_serialize_example)
print('serialized_features_dataset = \n{0}\n'.format(serialized_features_dataset))


def generator():
    for features in features_dataset:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
                                generator, output_types=tf.string, output_shapes=()
                            )

print('serialized_features_dataset = \n{0}\n'.format(serialized_features_dataset))

# And write them to a TFRecord file:
tf_record_path = PROJECT_ROOT_DIR.joinpath('tf_record')
filename = str(tf_record_path.joinpath('test.tfrecord'))
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Reading a TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
You can also read the TFRecord file using the tf.data.TFRecordDataset class.

More information on consuming TFRecord files using tf.data can be found here.

Using TFRecordDatasets can be useful for standardizing input data and optimizing performance.
---------------------------------------------------------------------------------------------------------------
'''
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('raw_dataset = \n{0}\n'.format(raw_dataset))

'''
---------------------------------------------------------------------------------------------------------------
At this point the dataset contains serialized tf.train.Example messages. 
When iterated over it returns these as scalar string tensors.

Use the .take method to only show the first 10 records.

Note: 
iterating over a tf.data.Dataset only works with eager execution enabled.
---------------------------------------------------------------------------------------------------------------
'''
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

'''
---------------------------------------------------------------------------------------------------------------
These tensors can be parsed using the function below. 
Note that the feature_description is necessary here because datasets use graph-execution, 
and need this description to build their shape and type signature:
---------------------------------------------------------------------------------------------------------------
'''
# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

'''
----------------------------------------------------------------------------------------------------------------
Alternatively, use tf.parse example to parse the whole batch at once. 
Apply this function to each item in the dataset using the tf.data.Dataset.map method:
----------------------------------------------------------------------------------------------------------------
'''
parsed_dataset = raw_dataset.map(_parse_function)
print('parsed_dataset = \n{0}\n'.format(parsed_dataset))

'''
----------------------------------------------------------------------------------------------------------------
Use eager execution to display the observations in the dataset. 
There are 10,000 observations in this dataset, but you will only display the first 10. 
The data is displayed as a dictionary of features. 
Each item is a tf.Tensor, and the numpy element of this tensor displays the value of the feature:
----------------------------------------------------------------------------------------------------------------
'''

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))

'''
---------------------------------------------------------------------------------------------------------------
Here, the tf.parse_example function unpacks the tf.Example fields into standard tensors.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       TFRecord files in Python                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
The tf.io module also contains pure-Python functions for reading and writing TFRecord files.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Writing a TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Next, write the 10,000 observations to the file test.tfrecord. 
Each observation is converted to a tf.Example message, then written to file. 
You can then verify that the file test.tfrecord has been created:
---------------------------------------------------------------------------------------------------------------
'''
# Write the `tf.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

file_capa = os.path.getsize(filename)
print('file name = {0}, file capa = {1}\n'.format(filename, file_capa))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Reading a TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
These serialized tensors can be easily parsed using tf.train.Example.ParseFromString:
---------------------------------------------------------------------------------------------------------------
'''
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('raw_dataset = \n{0}\n'.format(raw_dataset))

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       RWalkthrough: Reading and writing image data                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
This is an end-to-end example of how to read and write image data using TFRecords. Using an image as input data, 
you will write the data as a TFRecord file, then read the file back and display the image.

This can be useful if, for example, you want to use several models on the same input dataset. 
Instead of storing the image data raw, it can be preprocessed into the TFRecords format, 
and that can be used in all further processing and modelling.

First, let's download this image of a cat in the snow and this photo of the Williamsburg Bridge, NYC under construction.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Fetch the images                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

example_image_path = PROJECT_ROOT_DIR.joinpath('original_data', 'example_images')

cat_in_snow  = tf.keras.utils.get_file(
                    str(example_image_path.joinpath('320px-Felis_catus-cat_on_snow.jpg')), 
                    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
                )

williamsburg_bridge = tf.keras.utils.get_file(
                    str(example_image_path.joinpath('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')),
                    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'
                )

im = Image.open(str(example_image_path.joinpath('320px-Felis_catus-cat_on_snow.jpg')))
im.show()

im = Image.open(str(example_image_path.joinpath('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')))
im.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Write the TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
As before, encode the features as types compatible with tf.Example. 
This stores the raw image string feature, as well as the height, width, depth, and arbitrary label feature. 
The latter is used when you write the file to distinguish between the cat image and the bridge image. 
Use 0 for the cat image, and 1 for the bridge image:
-------------------------------------------------------------------------------------------------------------
'''
image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)

print('...')

'''
----------------------------------------------------------------------------------------------
Notice that all of the features are now stored in the tf.Example message. 
Next, functionalize the code above and write the example messages to a file named images.tfrecords:
----------------------------------------------------------------------------------------------
'''

# Write the raw image files to `images.tfrecords`.
# First, process the two images into `tf.Example` messages.
# Then, write to a `.tfrecords` file.
record_file = str(PROJECT_ROOT_DIR.joinpath('tf_record', 'images.tfrecords'))

with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

file_capa = os.path.getsize(record_file)
print('file name = {0}, file capa = {1}\n'.format(record_file, file_capa))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Read the TFRecord file                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
You now have the file—images.tfrecords—and can now iterate over the records in it to read back what you wrote. 
Given that in this example you will only reproduce the image, the only feature you will need is the raw image string. 
Extract it using the getters described above, namely example.features.feature['image_raw'].bytes_list.value[0]. 
You can also use the labels to determine which record is the cat and which one is the bridge:
-------------------------------------------------------------------------------------------------------------
'''
raw_image_dataset = tf.data.TFRecordDataset(record_file)

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print('parsed_image_dataset = \n{0}\n'.format(parsed_image_dataset))

# Recover the images from the TFRecord file:
for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    im = Image.open(image_raw)
    im.show()

data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print(
        '       finished       datasets_TFRecord_tf_example.py                       ({0})       \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()