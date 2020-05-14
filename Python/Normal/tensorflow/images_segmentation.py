'''
------------------------------------------------------------------------------------------
images
    TImage segmentation

This tutorial focuses on the task of image segmentation, using a modified U-Net.

What is image segmentation?

So far you have seen image classification, where the task of the network is to assign a label or class to an input image. 
However, suppose you want to know where an object is located in the image, the shape of that object, which pixel belongs to which object, etc. 
In this case you will want to segment the image, i.e., each pixel of the image is given a label. 
Thus, the task of image segmentation is to train a neural network to output a pixel-wise mask of the image. 
This helps in understanding the image at a much lower level, i.e., the pixel level. 
Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging to name a few.

The dataset that will be used for this tutorial is the Oxford-IIIT Pet Dataset, created by Parkhi et al. The dataset consists of images, 
their corresponding labels, and pixel-wise masks. The masks are basically labels for each pixel. Each pixel is given one of three categories :

	* Class 1 : Pixel belonging to the pet.
	* Class 2 : Pixel bordering the pet.
	* Class 3 : None of the above/ Surrounding pixel.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import pprint
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
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow_examples.models.pix2pix import pix2pix

print(__doc__)

keras = tf.keras
tfds.disable_progress_bar()

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

pprint.pprint(sys.path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the Oxford-IIIT Pets dataset                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The dataset is already included in TensorFlow datasets, all that is needed to do is download it. 
The segmentation masks are included in version 3.0.0, which is why this particular version is used.
---------------------------------------------------------------------------------------------------------------
'''

'''
dataset, info = tfds.load(
                    'oxford_iiit_pet:3.0.0', 
                    data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                    download=True,
                    with_info=True,
                    as_supervised=True
                )
'''
file_path = PROJECT_ROOT_DIR.joinpath('Data', 'oxford_iiit_pet', 'datasets', 'images')
dir_path = PROJECT_ROOT_DIR.joinpath('Data', 'oxford_iiit_pet')

dataset = tf.keras.utils.get_file(
            origin='http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
            fname= file_path,
            untar=True,
            extract=True,
            cache_dir=dir_path
        )

dataset_root = Path(dataset)
print('dataset_root = \n{0}\n'.format(dataset_root))

for item in dataset_root.iterdir():
    print(item)

image_count = len(list(dataset_root.glob('*.jpg')))
print('image_count = {0}\n'.format(image_count))

info = tf.keras.utils.get_file(
            origin='http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz',
            fname=PROJECT_ROOT_DIR.joinpath('Data', 'oxford_iiit_pet', 'datasets', 'annotations'), 
            untar=True,
            extract=True,
            cache_dir=PROJECT_ROOT_DIR.joinpath('Data', 'oxford_iiit_pet')
        )

info_root = Path(info)
print('info_root = \n{0}\n'.format(info_root))

for item in info_root.iterdir():
    if 'trainval.txt' in str(item):
        df = pd.read_table(item, header=None, delim_whitespace=True)
        item = str(item).replace('.txt','.csv')
        df.to_csv(item, index=False, header=['file name','class ids', 'species', 'breed id'])
    if 'test.txt' in str(item):
        df = pd.read_table(item, header=None, delim_whitespace=True)
        item = str(item).replace('.txt','.csv')
        df.to_csv(item, index=False, header=['file name','class ids', 'species', 'breed id'])
    
    print(item)

'''
----------------------------------------------------------------------------------------------------------------
The following code performs a simple augmentation of flipping an image. 
In addition, image is normalized to [0,1]. 
Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. 
For the sake of convinience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}.
----------------------------------------------------------------------------------------------------------------
'''

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

'''
------------------------------------------------------------------------------------------------------------
The dataset already contains the required splits of test and train and so let's continue to use the same split.
------------------------------------------------------------------------------------------------------------
'''
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE