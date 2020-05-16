'''
------------------------------------------------------------------------------------------
images
    Convolutional Neural Network (CNN)

This tutorial demonstrates training a simple Convolutional Neural Network (CNN) to classify CIFAR images. 
Because this tutorial uses the Keras Sequential API, 
creating and training our model will take just a few lines of code.
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

print(
        '----------------------------------------------------------------------------------------\n'
        '       Download and prepare the CIFAR10 dataset                                         \n'
        '----------------------------------------------------------------------------------------\n'
    )
'''
----------------------------------------------------------------------------------------------------------------
The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class. 
The dataset is divided into 50,000 training images and 10,000 testing images. 
The classes are mutually exclusive and there is no overlap between them.
----------------------------------------------------------------------------------------------------------------
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print   (
        '---------------------------------------------------------------------------------------\n'
        '       Verify the data                                                                 \n'
        '---------------------------------------------------------------------------------------\n'
        )
# To verify that the dataset looks correct, 
# let's plot the first 25 images from the training set and display the class name below each image.
class_names = [
                    'airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'
            ]

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.magma)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])

plt.show()

print   (
        '-------------------------------------------------------------------------------------\n'
        '       Create the convolutional base                                                 \n'
        '-------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. 
If you are new to these dimensions, color_channels refers to (R,G,B). 
In this example, you will configure our CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images. 
You can do this by passing the argument input_shape to our first layer.
--------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# Let's display the architecture of our model so far.
model.summary()

'''
----------------------------------------------------------------------------------------------------------------
Above, 
you can see that the output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels). 
The width and height dimensions tend to shrink as you go deeper in the network. 
The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64). 
Typically, as the width and height shrink, you can afford (computationally) to add more output channels in each Conv2D layer.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '-------------------------------------------------------------------------------------\n'
        '       Add Dense layers on top                                                       \n'
        '-------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To complete our model, 
you will feed the last output tensor from the convolutional base (of shape (3, 3, 64)) 
into one or more Dense layers to perform classification. 
Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. 
First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. 
CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs and a softmax activation.
---------------------------------------------------------------------------------------------------------------
'''
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

'''
--------------------------------------------------------------------------------------------------------------
Here's the complete architecture of our model.
--------------------------------------------------------------------------------------------------------------
'''
model.summary()

'''
-------------------------------------------------------------------------------------------------------------
As you can see, 
our (3, 3, 64) outputs were flattened into vectors of shape (576) before going through two Dense layers.
-------------------------------------------------------------------------------------------------------------
'''
print   (
        '--------------------------------------------------------------------------------------\n'
        '       Compile and train the model                                                    \n'
        '--------------------------------------------------------------------------------------\n'
        )
model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

history = model.fit(
                        train_images, 
                        train_labels, 
                        epochs=10, 
                        validation_data=(test_images, test_labels)
                    )

print   (
        '--------------------------------------------------------------------------------------\n'
        '       Evaluate the model                                                             \n'
        '--------------------------------------------------------------------------------------\n'
        )
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('test_loss = {0}, test_acc = {0}\n'.format(test_loss, test_acc))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished         images_cnn.py                          (2020/05/16)                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()