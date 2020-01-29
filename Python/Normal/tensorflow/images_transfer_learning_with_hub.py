'''
------------------------------------------------------------------------------------------
images
    Transfer learning with TensorFlow Hub

TensorFlow Hub is a way to share pretrained model components. 
See the TensorFlow Module Hub for a searchable listing of pre-trained models. 
This tutorial demonstrates:

	1. How to use TensorFlow Hub with tf.keras.
	2. How to do image classification using TensorFlow Hub.
	3. How to do simple transfer learning.
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

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
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
        '       An ImageNet classifier                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the classifier                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use hub.module to load a mobilenet, and tf.keras.layers.Lambda to wrap it up as a keras layer. 
Any TensorFlow 2 compatible image classifier URL from tfhub.dev will work here.
---------------------------------------------------------------------------------------------------------------
'''
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
                hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
            ])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Run it on a single image                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Download a single image to try the model on.
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper.show()

grace_hopper = np.array(grace_hopper)/255.0

print('grace_hopper.shape = {0}\n'.format(grace_hopper.shape))

# Add a batch dimension, and pass the image to the model.
result = classifier.predict(grace_hopper[np.newaxis, ...])

print('result.shape = {0}\n'.format(result.shape))

'''
-------------------------------------------------------------------------------------------------------------
The result is a 1001 element vector of logits, rating the probability of each class for the image.

So the top class ID can be found with argmax:
-------------------------------------------------------------------------------------------------------------
'''
predicted_class = np.argmax(result[0], axis=-1)

print('predicted_class = {0}\n'.format(predicted_class))

