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
        '       Load the image using tf.data.                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
data_root_orig = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        fname=PROJECT_ROOT_DIR.joinpath('original_data', 'datasets', 'flower_photos'), 
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

'''
--------------------------------------------------------------------------------------------------------------
Create a new dataset that loads and shapes the image at runtime 
by mapping preprocess_image to the dataset in the file path.
--------------------------------------------------------------------------------------------------------------
'''
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8,8))

for n,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Data set of (image, label) pair                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# You can create a label dataset using the same from_tensor_slices method.
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

# Output the first 10 labels.
for label in label_ds.take(10):
    print(label_names[label.numpy()])

# Since these datasets are in the same order, zipping creates a dataset of (image, label) pairs.
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# The new dataset's shapes and types are tuples of shapes and types indicating their fields.
print('image_label_ds = \n{0}\n'.format(image_label_ds))

'''
-------------------------------------------------------------------------------------------------------------------
Note: 
If you have arrays like all_image_labels or all_image_paths, 
an alternative to the tf.data.dataset.Dataset.zip method is to slice pairs of arrays.
-------------------------------------------------------------------------------------------------------------------
'''
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print('image_label_ds = \n{0}\n'.format(image_label_ds))

'''
---------------------------------------------------------------------------------------------------------------------
Basic training techniques

To train a model using this dataset, the data must be

* Well shuffled
* Batched
* Endlessly repeated
* Batches must be available as soon as possible.

These properties can easily be added using the tf.data API.
----------------------------------------------------------------------------------------------------------------------
'''
BATCH_SIZE = 32

# By setting the size of the shuffle buffer the same as the data set, 
# the data can be completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

# Using `prefetch` allows the dataset to fetch batches in the background while training the model.
ds = ds.prefetch(buffer_size=AUTOTUNE)
print('ds = \n{0}\n'.format(ds))

'''
-----------------------------------------------------------------------------------------------------------------------
There are a few things to keep in mind.

1. Order is important.

* .shuffle before .repeat shuffles elements across epoch boundaries. 
    (There may be some elements that appear twice before all other elements appear)

* .shuffle after .batch shuffles the order of the batch, but does not shuffle elements across batches.

2. For a complete shuffle, set buffer_size to the same size as the dataset. 
    If it is less than the size of the dataset, larger values ​​are better randomized, but use more memory.

3. Elements are fetched after the shuffle buffer is full. Therefore, 
    a large buffer_size causes a delay when starting to use the Dataset.

4. The shuffled dataset does not signal that the dataset is over until the shuffle buffer is completely empty. 
    When the .repeat restarts the Dataset, there is another wait until the shuffle buffer is full.

The last problem can be addressed by combining the tf.data.Dataset.apply method with the fused tf.data.experimental.shuffle_and_repeat function.
----------------------------------------------------------------------------------------------------------------------
'''
ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
    )

ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print('ds = \n{0}\n'.format(ds))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Connect dataset to model                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Obtain a copy of MobileNet v2 from tf.keras.applications.

Use this for a simple transfer learning sample.

Set MobileNet weights to non-trainable.
---------------------------------------------------------------------------------------------------------------
'''
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

'''
---------------------------------------------------------------------------------------------------------------
This model assumes that the inputs are normalized to the range [-1,1].

Therefore, before passing data to the MobileNet model, the input must be converted from the range [0,1] to the range [-1,1].
----------------------------------------------------------------------------------------------------------------
'''
def change_range(image,label):
    return 2*image-1, label

print('---< keras_ds >---\n')
keras_ds = ds.map(change_range)

print()

'''
-----------------------------------------------------------------------------------------------------------------
MobileNet returns a 6x6 feature space for each image.
-----------------------------------------------------------------------------------------------------------------
'''
# Let's give one batch.
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print('feature_map_batch.shape = {0}\n'.format(feature_map_batch.shape))

'''
------------------------------------------------------------------------------------------------------------------
Create a model that wraps MobileNet 
and calculate the average value along the axis of space with tf.keras.layers.GlobalAveragePooling2D 
before the output layer tf.keras.layers.Dense.
------------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(label_names))
    ])

'''
-------------------------------------------------------------------------------------------------------------------
The output of the expected shape is obtained.
-------------------------------------------------------------------------------------------------------------------
'''
print('---< The output of the expected shape is obtained. >---')
logit_batch = model(image_batch).numpy()

print('min logit: {0}'.format(logit_batch.min()))
print('max logit: {0}'.format(logit_batch.max()))
print()

print("Shape: {0}\n".format(logit_batch.shape))

# Compile the model to describe the training method.
model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

'''
----------------------------------------------------------------------------------------------------------------------
There are two trainable variables: weights and bias for the fully connected layer.
----------------------------------------------------------------------------------------------------------------------
'''
print('len(model.trainable_variables) = {0}\n'.format(len(model.trainable_variables)))

print('model.summary() = \n{0}\n'.format(model.summary()))

'''
----------------------------------------------------------------------------------------------------------------------
Train the model.

Normally you will specify the actual number of steps per epoch, 
but for the purpose of this demonstration we will only use 3 steps.
-----------------------------------------------------------------------------------------------------------------------
'''
print('---< Train the model. >---')
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

print('steps_per_epoch = {0}\n'.format(steps_per_epoch))

# fitting model
model.fit(ds, epochs=1, steps_per_epoch=3)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Performance                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A simple pipeline reads one file for each epoch. 
This is not a problem for local training with a CPU, 
but training with a GPU is not enough and should not be used for any distributed training.

To investigate, we first define a simple function to check the performance of the dataset.
---------------------------------------------------------------------------------------------------------------
'''
default_timeit_steps = 2*steps_per_epoch+1

def timeit(ds, steps=default_timeit_steps):
    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
    # before starting the timer
    it = iter(ds.take(steps+1))
    next(it)

    start = time.time()
    for i,(images,labels) in enumerate(it):
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
    print("Total time: {}s".format(end-overall_start))

'''
------------------------------------------------------------------------------------------------------------------
The performance of the current dataset is as follows:
------------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Performance                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
    )

ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print('ds = \n{0}\n'.format(ds))

timeit(ds)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       cache                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use tf.data.Dataset.cache to easily cache the calculation results across epochs. 
This is especially effective when the data fits in memory.

Here, the image is cached after being pre-processed (decoded and resized).
---------------------------------------------------------------------------------------------------------------
'''
print('---< first try >---')
ds = image_label_ds.cache()
ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
    )
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

print('ds = \n{0}\n'.format(ds))

timeit(ds)

'''
----------------------------------------------------------------------------------------------------------------
One of the disadvantages of using a memory cache is that the cache must be rebuilt each time it is executed. 
For this reason, each time a dataset is started, there is only a delay for starting.
----------------------------------------------------------------------------------------------------------------
'''
print('---< second try >---')

timeit(ds)

'''
---------------------------------------------------------------------------------------------------------------
If the data does not fit in memory, use a cache file.
---------------------------------------------------------------------------------------------------------------
'''
print('---< using cash file 1st try >---')
cash_path = str(PROJECT_ROOT_DIR.joinpath('cache', 'cache.tf-data'))

ds = image_label_ds.cache(filename=cash_path)
ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
    )
ds = ds.batch(BATCH_SIZE).prefetch(1)

print(ds)

timeit(ds)

print('---< using cash file 2nd try >---')
'''
---------------------------------------------------------------------------------------------------------------
Cache files also have the advantage that the dataset can be restarted without rebuilding the cache. 
Let's see how fast the second time is.
---------------------------------------------------------------------------------------------------------------
'''
timeit(ds)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       TFRecord file                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Raw image data

TFRecord files are a simple format for storing sequences of large binary objects. 
By packing multiple samples into the same file, TensorFlow can load multiple samples at once. 
This is especially important for performance when using remote storage services like GCS.
---------------------------------------------------------------------------------------------------------------
'''
# First, build a TFRecord file from the raw image data.
image_tfrec_path = str(PROJECT_ROOT_DIR.joinpath('images.tfrec'))

image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter(image_tfrec_path)
tfrec.write(image_ds)

# Next, read the TFRecord file and build a dataset to decode / reformat 
# the image using the preprocess_image function defined earlier.
image_ds = tf.data.TFRecordDataset(image_tfrec_path).map(preprocess_image)

# Zip this with the previously defined label dataset to get the expected (image, label) pair.
ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
    )

ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print('ds = \n{0}\n'.format(ds))

timeit(ds)

'''
---------------------------------------------------------------------------------------------------------------
This is slower than the cache version. This is because preprocessing is not cached.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Serialized tensor                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# To save preprocessing to a TFRecord file, create a dataset of preprocessed images as you did before.
paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)

print('image_ds = \n{0}\n'.format(image_ds))
'''
-----------------------------------------------------------------------------------------------------------------
This is a tensor dataset, not a .jpeg string dataset.
----------------------------------------------------------------------------------------------------------------
'''
# To serialize this to a TFRecord file, first convert the tensor dataset to a string dataset.
ds = image_ds.map(tf.io.serialize_tensor)

print('ds = \n{0}\n'.format(ds))

tfrec = tf.data.experimental.TFRecordWriter(image_tfrec_path)
tfrec.write(ds)

'''
-----------------------------------------------------------------------------------------------------------------
With preprocessing cached, data can be loaded very efficiently from TFRecord files. 
Remember to deserialize before using tensors.
-----------------------------------------------------------------------------------------------------------------
'''
ds = tf.data.TFRecordDataset(image_tfrec_path)

def parse(x):
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [192, 192, 3])
    return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

print('ds = \n{0}\n'.format(ds))

# Then add a label and apply the same standard processing as before.
ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print('ds = \n{0}\n'.format(ds))

timeit(ds)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        datasets_image.py                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()