'''
------------------------------------------------------------------------------------------
images
    Image classification

This tutorial shows how to classify cats or dogs from images. 
It builds an image classifier using a tf.keras.Sequential model 
and load data using tf.keras.preprocessing.image.ImageDataGenerator. 
You will get some practical experience and develop intuition for the following concepts:

	* Building data input pipelines using the tf.keras.preprocessing.image.ImageDataGenerator class 
	to efficiently work with data on disk to use with the model.
	
	* Overfitting —How to identify and prevent it.
	
	* Data augmentation and dropout —Key techniques to fight overfitting in computer vision tasks 
	to incorporate into the data pipeline and image classifier model.
	
This tutorial follows a basic machine learning workflow:

	1. Examine and understand data
	2. Build an input pipeline
	3. Build the model
	4. Train the model
	5. Test the model
	6. Improve the model and repeat the process

Import packages

Let's start by importing the required packages. 
The os package is used to read files and directory structure, 
NumPy is used to convert python list to numpy array and to perform required matrix operations and matplotlib.pyplot 
to plot the graph and display images in the training and validation data.
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        '       Download and prepare the CIFAR10 dataset                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

fName = PROJECT_ROOT_DIR.joinpath('Data/cats_and_dog.zip')

cacheDir = PROJECT_ROOT_DIR.joinpath('Data')

path_to_zip = tf.keras.utils.get_file(
                                        origin=_URL,
                                        fname=fName, 
                                        extract=True,
                                        cache_dir=cacheDir
                                    )

PATH = os.path.join(os.path.dirname(path_to_zip), 'datasets\\cats_and_dogs_filtered')

'''
--------------------------------------------------------------------------------------------------------------
The dataset has the following directory structure:

cats_and_dogs_filtered
|__ train
    |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
    |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
|__ validation
    |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
After extracting its contents, assign variables with the proper file path for the training and validation set.
-----------------------------------------------------------------------------------------------------------------
'''
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Understand the data                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Let's look at how many cats and dogs images are in the training and validation directory:
--------------------------------------------------------------------------------------------------------------
'''
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

'''
------------------------------------------------------------------------------------------------------------------
For convenience, set up variables to use while pre-processing the dataset and training the network.
------------------------------------------------------------------------------------------------------------------
'''
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

