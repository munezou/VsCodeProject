'''
------------------------------------------------------------------------------------------
tf.data

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

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/python/basic')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dataset structure                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
tf_random_uniform = tf.random.uniform([4, 10])

for data in tf_random_uniform:
    print(data.numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf_random_uniform)
print('datasets1.element_spec = {0}\n'.format(dataset1.element_spec))

dataset2 = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([4]),
        tf.random.uniform([4, 100], 
        maxval=100, 
        dtype=tf.int32))
    )
print('datasets2.element_spec = {0}\n'.format(dataset2.element_spec))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print('dataset3.element_spec = {0}\n'.format(dataset3))

dataset = tf.data.Dataset.from_tensor_slices(
        {"a": tf.random.uniform([4]),
        "b": tf.random.uniform([4, 100], 
        maxval=100, 
        dtype=tf.int32)}
    )
print('dataset.element_spec = {0}\n'.format(dataset.element_spec))

print('---< map >---')
dataset1 = dataset1.map(lambda x: x + 1)
for data in dataset1:
    print(data.numpy())
    
print()
print('---< flat_map >---')
dataset4 = tf.data.Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
dataset4 = dataset4.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x + 1))
for data in dataset4:
    print(data.numpy())
print()

print('---< filter >---')
d = tf.data.Dataset.from_tensor_slices([1, 2, 3])

d = d.filter(lambda x: x < 3)  # ==> [1, 2]
print(list(d))

for data in d:
    print(data.numpy())

print()
# `tf.math.equal(x, y)` is required for equality comparison
def filter_fn(x):
    return tf.math.equal(x, 1)

d = d.filter(filter_fn)  # ==> [1]
for data in d:
    print(data.numpy())
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load the image using tf.data.                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
data_root_orig = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        fname=PROJECT_ROOT_DIR.joinpath('original_data/flower_photos'), 
        untar=True,
        cache_dir=PROJECT_ROOT_DIR.joinpath('original_data')
    )
data_root = Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*.jpg'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('image_count = {0}\n'.format(image_count))

print('all_image_paths[:10] = \n{0}\n'.format(all_image_paths[:10]))

attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = Path(image_path).relative_to(data_root)
    s = str(image_rel).split('\\')
    output_print = "Image (CC BY 2.0) By " + s[0]
    return output_print

for n in range(3):
    image_path = random.choice(all_image_paths)
    im = Image.open(os.path.join(image_path))
    im.show()
    print(caption_image(image_path))
    print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Determine the label for each image                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Output a list of labels.
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print('label_names = \n{0}\n'.format(label_names))

# Assign an index to the label.
label_to_index = dict((name, index) for index,name in enumerate(label_names))
print('label_to_index = \n{0}\n'.format(label_to_index))

# Create a list of file and label indices.
all_image_labels = [label_to_index[Path(path).parent.name] for path in all_image_paths]
print('First 10 labels indices: = {0}\n'.format(all_image_labels))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Loading and shaping images                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
img_path = all_image_paths[0]
print('img_path = \n{0}\n'.format(image_path))

img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")