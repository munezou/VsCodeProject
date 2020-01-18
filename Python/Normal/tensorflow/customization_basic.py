'''
------------------------------------------------------------------------------------------
customization
    Basic

This is an introductory TensorFlow tutorial shows how to:

Import the required package
* Create and use tensors
* Use GPU acceleration
* Demonstrate tf.data.Dataset
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
from pathlib import Path
from packaging import version
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

print(__doc__)

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
        '       Tensors                                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
A Tensor is a multi-dimensional array. 
Similar to NumPy ndarray objects, tf.Tensor objects have a data type and a shape. 
Additionally, tf.Tensors can reside in accelerator memory (like a GPU). 
TensorFlow offers a rich library of operations (tf.add, tf.matmul, tf.linalg.inv etc.) that consume and produce tf.Tensors. 
These operations automatically convert native Python types, for example:
--------------------------------------------------------------------------------------------------------------
'''
print('tf.add(1, 2) = ', tf.add(1, 2))
print()
print('tf.add([1, 2], [3, 4]) = ', tf.add([1, 2], [3, 4]))
print()
print('tf.square(5) = ', tf.square(5))
print()
print('tf.reduce_sum([1, 2, 3]) = ', tf.reduce_sum([1, 2, 3]))
print()

# Operator overloading is also supported
print('tf.square(2) + tf.square(3) = ', tf.square(2) + tf.square(3))
print()

'''
---------------------------------------------------------------------------------------------------------------
Each tf.Tensor has a shape and a datatype:
---------------------------------------------------------------------------------------------------------------
'''
x = tf.matmul([[1]], [[2, 3]])
print('x = {0}\n'.format(x))
print('x.shape = {0}\n'.format(x.shape))
print('x.dtype = {0}\n'.format(x.dtype))

'''
----------------------------------------------------------------------------------------------------------------
The most obvious differences between NumPy arrays and tf.Tensors are:

1. Tensors can be backed by accelerator memory (like GPU, TPU).
2. Tensors are immutable.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       NumPy Compatibility                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Converting between a TensorFlow tf.Tensors and a NumPy ndarray is easy:

* TensorFlow operations automatically convert NumPy ndarrays to Tensors.
* NumPy operations automatically convert Tensors to NumPy ndarrays.

Tensors are explicitly converted to NumPy ndarrays using their .numpy() method. 
These conversions are typically cheap since the array and tf.Tensor share the underlying memory representation, if possible. 
However, sharing the underlying representation isn't always possible since the tf.Tensor may be hosted in GPU memory 
while NumPy arrays are always backed by host memory, and the conversion involves a copy from GPU to host memory.
-----------------------------------------------------------------------------------------------------------------
'''

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)
print()

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))
print()

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       GPU acceleration                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Many TensorFlow operations are accelerated using the GPU for computation. 
Without any annotations, 
TensorFlow automatically decides whether to use the GPU or CPU for an operation—copying the tensor between CPU and GPU memory, if necessary. 
Tensors produced by an operation are typically backed by the memory of the device on which the operation executed, for example:
---------------------------------------------------------------------------------------------------------------
'''
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       GPU acceleration                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Many TensorFlow operations are accelerated using the GPU for computation. 
Without any annotations, 
TensorFlow automatically decides whether to use the GPU or CPU for an operation—copying the tensor between CPU and GPU memory, if necessary. 
Tensors produced by an operation are typically backed by the memory of the device on which the operation executed, for example:
----------------------------------------------------------------------------------------------------------------
'''

x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Device Names                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
The Tensor.device property provides a fully qualified string name of the device hosting the contents of the tensor. 
This name encodes many details, 
such as an identifier of the network address of the host on which this program is executing and the device within that host. 
This is required for distributed execution of a TensorFlow program. 
The string ends with GPU:<N> if the tensor is placed on the N-th GPU on the host.
---------------------------------------------------------------------------------------------------------------
'''

