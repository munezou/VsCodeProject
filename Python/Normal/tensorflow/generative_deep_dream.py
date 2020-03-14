'''
------------------------------------------------------------------------------------------
genarative
    DeepDream

This tutorial contains a minimal implementation of DeepDream, as described in this blog post by Alexander Mordvintsev.

DeepDream is an experiment that visualizes the patterns learned by a neural network. 
Similar to when a child watches clouds and tries to interpret random shapes, 
DeepDream over-interprets and enhances the patterns it sees in an image.

It does so by forwarding an image through the network, 
then calculating the gradient of the image with respect to the activations of a particular layer. 
The image is then modified to increase these activations, 
enhancing the patterns seen by the network, and resulting in a dream-like image. 
This process was dubbed "Inceptionism" (a reference to InceptionNet, and the movie Inception.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
import tempfile
import functools
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
pd.options.display.max_rows = None
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

'''
------------------------------------------------------------------------------------------
Let's demonstrate how you can make a neural network "dream" and enhance the surreal patterns it sees in an image.
------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/dogception.png'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Choose an image to dream-ify                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
For this tutorial, let's use an image of a labrador.
------------------------------------------------------------------------------------------
'''
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# Download an image and read it into a NumPy array.
def download(url, target_size=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    return img

# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


# Display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# Downsizing the image makes it easier to work with.
original_img = download(url, target_size=[225, 375])
original_img = np.array(original_img)

show(original_img)

