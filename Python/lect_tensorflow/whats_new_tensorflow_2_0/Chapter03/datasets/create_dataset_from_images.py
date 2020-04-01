'''
------------------------------------------------------------------------------------------
chapter03/datasets
    Illustration of tf.data.Dataset.from_tensors(...) and tf.data.Dataset.from_tensor_slices(...) APIs
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
from packaging import version

import numpy as np

import tensorflow as tf

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

def preprocess_image(img_path, label):
    img_data = tf.io.read_file(img_path)
    feat = tf.image.decode_jpeg(img_data, channels=3)
    feat = tf.image.convert_image_dtype(feat, tf.float32)
    return feat, label, img_path

def get_label(img_path):
    if isinstance(img_path, bytes):
        img_path = img_path.decode(sys.getdefaultencoding())
    fn = os.path.basename(img_path)
    cl = fn.split('_')[0]
    if cl == 'cat':
        label = 0
    else:
        label = 1

    return label

def read_img_file(img_path):
    label = get_label(img_path)
    return img_path, label

try:
    file_pattern = [os.path.join(PROJECT_ROOT_DIR, "curated_data/images/*.jpeg"), os.path.join(PROJECT_ROOT_DIR, "curated_data/images/*.jpg")]
    image_files = tf.io.gfile.glob(file_pattern)
    labels = []
    for img_path in image_files:
        labels.append(get_label(img_path))
except Exception as ex:
    print(ex)
    pass

print('image_files = \n{0}'.format(image_files))
print('labels = \n{0}\n'.format(labels))

image_path_dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))

print('image_path_dataset = \n{0}\n'.format(image_path_dataset))

image_dataset = image_path_dataset.map(preprocess_image)

print('image_dataset = \n{0}\n'.format(image_dataset))