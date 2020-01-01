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

print('---< map >---')
'''
-----------------------------------------------------------------------------------------------------------------
map(
    map_func,
    num_parallel_calls=None
)

Maps map_func across the elements of this dataset.

This transformation applies map_func to each element of this dataset, 
and returns a new dataset containing the transformed elements, 
in the same order as they appeared in the input.
------------------------------------------------------------------------------------------------------------------
'''
a = tf.data.Dataset.range(1, 6)
a1 = a.map(lambda x : x + 1)
for data in a1:
    print(data.numpy())
    
# The input signature of map_func is determined by the structure of each element in this dataset. For example:
# NOTE: The following examples use `{ ... }` to represent the
# contents of a dataset.
# Each element is a `tf.Tensor` object.
a2 = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])
print('a = {0}, '.format(a2))

# `map_func` takes a single argument of type `tf.Tensor` with the same
# shape and dtype.
a2 = a2.map(lambda x: tf.math.sqrt(x))
print('a2.map(lambda x: tf.math.sqrt(x)) = \n{0}\n'.format(a2))

for data in a2:
    print(data.numpy())

'''
# Each element is a tuple containing two `tf.Tensor` objects.
b = tf.data.Dataset(tf.tuple((1, "foo"), (2, "bar"), (3, "baz")))
print('b = \n{0}\n'.format(b))

for data in b:
    print(data)

# `map_func` takes two arguments of type `tf.Tensor`.
result = b.map(lambda x_int, y_str: (x_int + 1, y_str))
print('b.map(lambda x_int, y_str: ...) = \n{0}\n'.format(result))


# Each element is a dictionary mapping strings to `tf.Tensor` objects.
c = { {"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"} }
print('c = \n{0}\n'.format(c))

# `map_func` takes a single argument of type `dict` with the same keys as
# the elements.
result = c.map(lambda d: ...)
print('c.map(lambda d: ...) = \n{0}\n'.format(result))
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load csv file using tf.data.                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file(
        origin=TRAIN_DATA_URL,
        fname=PROJECT_ROOT_DIR.joinpath('original_data/titanic/train.csv')
    )


test_file_path = tf.keras.utils.get_file(
        origin=TEST_DATA_URL,
        fname=PROJECT_ROOT_DIR.joinpath('original_data/titanic/eval.csv')
    )

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# output first row
with open(train_file_path, 'r') as f:
    names_row = f.readline()

CSV_COLUMNS = names_row.rstrip('\n').split(',')
print('CSV_COLUMNS = \n{0}\n'.format(CSV_COLUMNS))

# You need to identify the column that will be the label for each sample and indicate what it is.
LABELS = [0, 1]
LABEL_COLUMN = 'survived'

FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=12, # It is set small to make it easier to see.
    label_name=LABEL_COLUMN,
    na_value="?",
    num_epochs=1,
    ignore_errors=True)
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

'''
-----------------------------------------------------------------------------------------------------
The elements that make up the dataset are batches represented as tuples of the form 
(multiple samples, multiple labels). 
The data in the sample is organized as column-based tensors (rather than row-based tensors), 
each containing a batch-sized (12 in this case) component.
-----------------------------------------------------------------------------------------------------
'''
examples, labels = next(iter(raw_train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Preprocessing of data.                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------------
Category data
Some columns in this CSV data are category columns. 
That is, its content must be one of a limited set of options.

In this CSV, these choices are represented as text. 
This text needs to be converted to numbers so that you can train the model. 
To make this easier, you need to create a list of category columns and a list of their choices.
---------------------------------------------------------------------------------------------------------
'''
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

'''
----------------------------------------------------------------------------------------------------------
Write a function that takes a categorical value tensor, matches it with a list of value names, 
and then performs one-hot encoding.
----------------------------------------------------------------------------------------------------------
'''



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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Decode to image tensor.                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
img_tensor = tf.image.decode_image(img_raw)

print('img_tensor.shape = {0}\n'.format(img_tensor.shape))
print('img_tensor.dtype = {0}\n'.format(img_tensor.dtype))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Resize to fit your model.                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0

print('img_final.shape = {0}\n'.format(img_final.shape))
print('img_final.numpy().min() = {0}\n'.format(img_final.numpy().min()))
print('img_final.numpy().max() = {0}\n'.format(img_final.numpy().max()))

# Here's a simple function for later use.
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
plt.show()
print()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Construction of tf.data.Dataset                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Image dataset

The easiest way to construct a tf.data.Dataset is to use the from_tensor_slices method.

Slicing an array of strings creates a string data set.
---------------------------------------------------------------------------------------------------------------
'''
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

'''
---------------------------------------------------------------------------------------------------------------
shapes and types indicate the contents of each item in the dataset. 
In this case, it is a set of scalars of binary strings.
--------------------------------------------------------------------------------------------------------------
'''
print('path_ds = \n{0}\n'.format(path_ds))
