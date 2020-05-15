'''
------------------------------------------------------------------------------------------
customization
    Custom layers
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
import tempfile

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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Custom layers                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
We recommend using tf.keras as a high-level API for building neural networks. 
That said, most TensorFlow APIs are usable with eager execution.
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Layers: common sets of useful operations                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Most of the time when writing code for machine learning models you want to operate at a higher level of abstraction 
than individual operations and manipulation of individual variables.

Many machine learning models are expressible as the composition and stacking of relatively simple layers, 
and TensorFlow provides both a set of many common layers as a well as easy ways for you to write your own application-specific layers 
either from scratch or as the composition of existing layers.

TensorFlow includes the full Keras API in the tf.keras package, and the Keras layers are very useful when building your own models.
----------------------------------------------------------------------------------------------------------------
'''
# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

'''
----------------------------------------------------------------------------------------------------------------
The full list of pre-existing layers can be seen in the documentation. 
It includes Dense (a fully-connected layer), Conv2D, LSTM, BatchNormalization, Dropout, and many others.
----------------------------------------------------------------------------------------------------------------
'''
# To use a layer, simply call it.
print('layer(tf.zeros([10, 5])) = \n{0}\n'.format(layer(tf.zeros([10, 5]))))

# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
print('layer.variables = \n{0}\n'.format(layer.variables))

# The variables are also accessible through nice accessors
print('layer.kernel = \n{0}\n'.format(layer.kernel))
print('layer.bias = \n{0}\n'.format(layer.bias))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Implementing custom layers                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
The best way to implement your own layer is extending the tf.keras.Layer class and implementing:

* __init__ , where you can do all input-independent initialization
* build, where you know the shapes of the input tensors and can do the rest of the initialization
* call, where you do the forward computation

Note that you don't have to wait until build is called to create your variables, you can also create them in __init__. 
However, 
the advantage of creating them in build is that it enables late variable creation based on the shape of the inputs the layer will operate on. 
On the other hand, creating variables in __init__ would mean that shapes required to create the variables will need to be explicitly specified.
----------------------------------------------------------------------------------------------------------------
'''
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable(
                                            "kernel",
                                            shape=[int(input_shape[-1]),
                                            self.num_outputs]
                                        )

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)

_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.

print([var.name for var in layer.trainable_variables])

'''
--------------------------------------------------------------------------------------------------------------------
Overall code is easier to read and maintain if it uses standard layers whenever possible, 
as other readers will be familiar with the behavior of standard layers. 
If you want to use a layer which is not present in tf.keras.layers, consider filing a github issue or, even better, 
sending us a pull request!
---------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Models: Composing layers                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------------------------------
Many interesting layer-like things in machine learning models are implemented by composing existing layers. 
For example, each residual block in a resnet is a composition of convolutions, batch normalizations, and a shortcut. 
Layers can be nested inside other layers.

Typically you inherit from keras.Model when you need the model methods like: Model.fit,Model.evaluate, and Model.save 
(see Custom Keras layers and models for details).

One other feature provided by keras.Model (instead of keras.layers.Layer) is that in addition to tracking variables, 
a keras.Model also tracks its internal layers, making them easier to inspect.

For example here is a ResNet block:
--------------------------------------------------------------------------------------------------------------------
'''
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

_ = block(tf.zeros([1, 2, 3, 3])) 

print('block.layer = \n{0}\n'.format(block.layers))

print('len(block.variables) = {0}\n'.format(len(block.variables)))

print('block.summary() = \n{0}\n'.format(block.summary()))

'''
------------------------------------------------------------------------------------------------------
Much of the time, however, models which compose many layers simply call one layer after the other. 
This can be done in very little code using tf.keras.Sequential:
-----------------------------------------------------------------------------------------------------
'''
my_seq = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Conv2D(2, 1, padding='same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Conv2D(3, (1, 1)),
                                tf.keras.layers.BatchNormalization()
                            ])

print('my_seq(tf.zeros([1, 2, 3, 3])) = \n{0}\n'.format(my_seq(tf.zeros([1, 2, 3, 3]))))

my_seq.summary()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished         customization_custom_layers.py                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()