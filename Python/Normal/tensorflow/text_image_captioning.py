'''
------------------------------------------------------------------------------------------
text
    Image captioning with visual attention

Given an image like the example below, 
our goal is to generate a caption such as "a surfer riding on a wave".
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
import pickle
from pathlib import Path
from packaging import version
from glob import glob
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/Surfing_in_Hawaii.jpg'))
im.show()
'''
---------------------------------------------------------------------------------------------------------------
To accomplish this, you'll use an attention-based model, 
which enables us to see what parts of the image the model focuses on as it generates a caption.
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/imcap_prediction.png'))
im.show()

'''
--------------------------------------------------------------------------------------------------------------
The model architecture is similar to Show, 
Attend and Tell: Neural Image Caption Generation with Visual Attention.

This notebook is an end-to-end example. 
When you run the notebook, it downloads the MS-COCO dataset, 
preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, 
and generates captions on new images using the trained model.

In this example, 
you will train a model on a relatively small amount of data—the first 30,000 captions for about 20,000 images 
(because there are multiple captions per image in the dataset).
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      Download and prepare the MS-COCO dataset                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
You will use the MS-COCO dataset to train our model. 
The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. 
The code below downloads and extracts the dataset automatically.

Caution: 
large download ahead. You'll use the training set, which is a 13GB file.
--------------------------------------------------------------------------------------------------------------
'''
annotation_zip = tf.keras.utils.get_file(
                    fname=str(PROJECT_ROOT_DIR.joinpath('Data/caption_data/captions.zip')),
                    cache_subdir=str(PROJECT_ROOT_DIR.joinpath('Data/caption_data')),
                    origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                    extract = True
                )

annotation_file = str(PROJECT_ROOT_DIR.joinpath('Data/caption_data/annotations/captions_train2014.json'))

name_of_zip = str(PROJECT_ROOT_DIR.joinpath('Data/caption_data/train2014.zip'))

if not os.path.exists(name_of_zip):
    image_zip = tf.keras.utils.get_file(
                    fname=name_of_zip,
                    cache_subdir=str(PROJECT_ROOT_DIR.joinpath('Data/caption_data')),
                    origin = 'http://images.cocodataset.org/zips/train2014.zip',
                    extract = True
                )

PATH = str(PROJECT_ROOT_DIR.joinpath('Data/caption_data/train2014/'))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      Optional: limit the size of the training set                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To speed up training for this tutorial, 
you'll use a subset of 30,000 captions and their corresponding images to train our model. 
Choosing to use more data would result in improved captioning quality.
---------------------------------------------------------------------------------------------------------------
'''
# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(
                                    all_captions,
                                    all_img_name_vector,
                                    random_state=1
                                )

# Select the first 30000 captions from the shuffled set
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

print('len(train_captions) = {0}, len(all_captions) = {1}\n'.format(len(train_captions), len(all_captions)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '      Preprocess the images using InceptionV3                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Next, you will use InceptionV3 (which is pretrained on Imagenet) to classify each image. 
You will extract features from the last convolutional layer.

First, you will convert the images into InceptionV3's expected format by:

	* Resizing the image to 299px by 299px
	* Preprocess the images using the preprocess_input method to normalize the image 
	so that it contains pixels in the range of -1 to 1, 
	which matches the format of the images used to train InceptionV3.
-------------------------------------------------------------------------------------------------------------
'''
