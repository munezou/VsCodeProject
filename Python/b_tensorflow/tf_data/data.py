'''
----------------------------------------------------------------------------------------------
tf.data: Build TensorFlow input pipelines

overview)

The tf.data API enables you to build complex input pipelines from simple, reusable pieces. For example, 
the pipeline for an image model might aggregate data from files in a distributed file system, 
apply random perturbations to each image, and merge randomly selected images into a batch for training. 
The pipeline for a text model might involve extracting symbols from raw text data, 
converting them to embedding identifiers with a lookup table, and batching together sequences of different lengths. 
The tf.data API makes it possible to handle large amounts of data, 
read from different data formats, and perform complex transformations.

The tf.data API introduces a tf.data.Dataset abstraction that represents a sequence of elements, 
in which each element consists of one or more components. 
For example, in an image pipeline, an element might be a single training example, 
with a pair of tensor components representing the image and its label.

There are two distinct ways to create a dataset:

A data source constructs a Dataset from data stored in memory or in one or more files.

A data transformation constructs a dataset from one or more tf.data.Dataset objects.
----------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import platform
import shutil
import subprocess
from packaging import version
from PIL import Image
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

pd.options.display.max_columns = None

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

np.set_printoptions(precision=4)

import urllib
proxy_support = urllib.request.ProxyHandler({'https': 'http://proxy.kanto.sony.co.jp:10080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Basic mechanics                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To create an input pipeline, you must start with a data source. 
For example, to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). 
Alternatively, if your input data is stored in a file in the recommended TFRecord format, you can use tf.data.TFRecordDataset().

Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. 
For example, you can apply per-element transformations such as Dataset.map(), 
and multi-element transformations such as Dataset.batch(). 
See the documentation for tf.data.Dataset for a complete list of transformations.

The Dataset object is a Python iterable. This makes it possible to consume its elements using a for loop:
----------------------------------------------------------------------------------------------------------------
'''
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print('dataset = \n{0}\n'.format(dataset))

for elem in dataset:
  print('elem = {0}'.format(elem.numpy()))

print()

# Or by explicitly creating a Python iterator using iter and consuming its elements using next:
it = iter(dataset)

for _ in range(10):
      try:
        print('next(it).numpy() = {0}'.format(next(it).numpy()))
        pass
      except Exception as ex:
        print(ex)
        pass
      finally:
        pass


print()

'''
------------------------------------------------------------------------------------------------------------------
Alternatively, 
dataset elements can be consumed using the reduce transformation, which reduces all elements to produce a single result. 
The following example illustrates how to use the reduce transformation to compute the sum of a dataset of integers.
------------------------------------------------------------------------------------------------------------------
'''
Sum_of_addition = dataset.reduce(0, lambda state, value: state + value).numpy()
print('Sum_of_addition = {0}\n'.format(Sum_of_addition))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dataset structure                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------------------------------
A dataset contains elements that each have the same (nested) structure and the individual components of the structure
 can be of any type representable by tf.TypeSpec, including Tensor, SparseTensor, RaggedTensor, TensorArray, or Dataset.

The Dataset.element_spec property allows you to inspect the type of each element component. 
The property returns a nested structure of tf.TypeSpec objects, matching the structure of the element, 
which may be a single component, a tuple of components, or a nested tuple of components. For example:
-----------------------------------------------------------------------------------------------------------------
'''
print('---< dataset1 >---')
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

print('dataset1.element_spec = \n{0}\n'.format(dataset1.element_spec))

for elem in dataset1:
  print('elem = {0}'.format(elem.numpy()))

print()

print('---< dataset2 >---')
dataset2 = tf.data.Dataset.from_tensor_slices((
    tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)
    ))

print('dataset2.element_spec = \n{0}\n'.format(dataset2.element_spec))

dataset2_elm = pd.DataFrame(dataset2)
print('dataset2_elm = \n{0}\n'.format(dataset2_elm))

print()

print('---< dataset3 >---')
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print('dataset3.element_spec = \n{0}\n'.format(dataset3.element_spec))

dataset3_elm = pd.DataFrame(dataset3)
print('dataset3_elm = \n{0}\n'.format(dataset3_elm))

print()

print('---< dataset4 >---')
# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

print('dataset4.element_spec = \n{0}\n'.format(dataset4.element_spec))

dataset4_elm = pd.DataFrame(dataset4)
print('dataset4_elm = \n{0}\n'.format(dataset4_elm))

print()

tf.SparseTensorSpec(tf.TensorShape([3, 4]), tf.int32)

# Use value_type to see the type of value represented by the element spec
print('dataset4.element_spec.value_type = \n{0}\n'.format(dataset4.element_spec.value_type))

'''
-----------------------------------------------------------------------------------------------------------------------------
The Dataset transformations support datasets of any structure. 
When using the Dataset.map(), and Dataset.filter() transformations, which apply a function to each element, 
the element structure determines the arguments of the function:
-----------------------------------------------------------------------------------------------------------------------------
'''
dataset1 = tf.data.Dataset.from_tensor_slices(
  tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

print('dataset1 = \n{0}\n'.format(dataset1))

for z in dataset1:
  print(z.numpy())

print()

dataset2 = tf.data.Dataset.from_tensor_slices((
  tf.random.uniform([4]),
  tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)
  ))

print('dataset2 = \n{0}\n'.format(dataset2))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print('dataset3 = \n{0}\n'.format(dataset3))

for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Reading input data                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Consuming NumPy arrays                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
If all of your input data fits in memory, 
he simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().
---------------------------------------------------------------------------------------------------------------
'''
train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
print('dataset = \n{0}\n'.format(dataset))

'''
---------------------------------------------------------------------------------------------------------------
Note: 
The above code snippet will embed the features and labels arrays in your TensorFlow graph as tf.constant() operations. 
This works well for a small dataset, 
but wastes memory---because the contents of the array will be copied multiple times---and can run into the 2GB limit for the tf.GraphDef protocol buffer.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Consuming Python generators                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Another common data source that can easily be ingested as a tf.data.Dataset is the python generator.

Caution: 
While this is a convienient approach it has limited portability and scalibility. 
It must run in the same python process that created the generator, and is still subject to the Python GIL.
---------------------------------------------------------------------------------------------------------------
'''
def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1

for n in count(5):
  print(n)

print()

'''
---------------------------------------------------------------------------------------------------------------
The Dataset.from_generator constructor converts the python generator to a fully functional tf.data.Dataset.

The constructor takes a callable as input, not an iterator. 
This allows it to restart the generator when it reaches the end. It takes an optional args argument, 
hich is passed as the callable's arguments.

The output_types argument is required because tf.data builds a tf.Graph internally, and graph edges require a tf.dtype.
--------------------------------------------------------------------------------------------------------------
'''
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

# batch(clumn) = 10, take(row) = 20
for count_batch in ds_counter.repeat().batch(10).take(20):
  print(count_batch.numpy())

'''
--------------------------------------------------------------------------------------------------------------
The output_shapes argument is not required but is highly recomended as many tensorflow operations do not support tensors with unknown rank. 
If the length of a particular axis is unknown or variable, set it as None in the output_shapes.

It's also important to note that the output_shapes and output_types follow the same nesting rules as other dataset methods.

Here is an example generator that demonstrates both aspects, it returns tuples of arrays, 
where the second array is a vector with unknown length.
--------------------------------------------------------------------------------------------------------------
'''
# Here is an example generator that demonstrates both aspects, 
# it returns tuples of arrays, where the second array is a vector with unknown length.
def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1

for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

'''
---------------------------------------------------------------------------------------------------------------
The first output is an int32 the second is a float32.
---------------------------------------------------------------------------------------------------------------
'''

# The first item is a scalar, shape (), and the second is a vector of unknown length, shape (None,)
ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,))
  )

print('ds_series  = \n{0}\n'.format(ds_series))

# Now it can be used like a regular tf.data.Dataset. 
# Note that when batching a dataset with a variable shape, you need to use Dataset.padded_batch.
ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=([], [None]))

ids, sequence_batch = next(iter(ds_series_batch))

print('ids.numpy() = \n{0}\n'.format(ids.numpy()))

print('sequence_batch.numpy() = \n{0}\n'.format(sequence_batch.numpy()))

'''
-------------------------------------------------------------------------------------------------------------
For a more realistic example, try wrapping preprocessing.image.ImageDataGenerator as a tf.data.Dataset.
-------------------------------------------------------------------------------------------------------------
'''
# First download the data:
flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True
  )

# Create the image.ImageDataGenerator
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

images, labels = next(img_gen.flow_from_directory(flowers))

print('images.dtype = {0}, images.shape = {1}\n'.format(images.dtype, images.shape))
print('labels.dtype = {0}, labels.shape = {1}\n'.format(labels.dtype, labels.shape))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Consuming TFRecord data                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The tf.data API supports a variety of file formats so that you can process large datasets that do not fit in memory. 
For example, 
the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data.
The tf.data.TFRecordDataset class enables you to stream over the contents of one or more TFRecord files as part of an input pipeline.
--------------------------------------------------------------------------------------------------------------
'''

# Here is an example using the test file from the French Street Name Signs (FSNS).
# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file(
    "fsns.tfrec", 
    "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
  )

'''
--------------------------------------------------------------------------------------------------------------
The filenames argument to the TFRecordDataset initializer can either be a string, a list of strings, or a tf.Tensor of strings. 
Therefore if you have two sets of files for training and validation purposes, 
you can create a factory method that produces the dataset, taking filenames as an input argument:
--------------------------------------------------------------------------------------------------------------
'''
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])

print('dataset = \n{0}\n'.format(dataset))

'''
--------------------------------------------------------------------------------------------------------------
Many TensorFlow projects use serialized tf.train.Example records in their TFRecord files. 
These need to be decoded before they can be inspected:
--------------------------------------------------------------------------------------------------------------
'''
raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed_features = parsed.features.feature['image/text']
print('parsed_features = \n{0}\n'.format(parsed_features))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Consuming text data                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Many datasets are distributed as one or more text files. 
The tf.data.TextLineDataset provides an easy way to extract lines from one or more text files. 
Given one or more filenames, a TextLineDataset will produce one string-valued element per line of those files.
---------------------------------------------------------------------------------------------------------------
'''
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

dataset = tf.data.TextLineDataset(file_paths)

# Here are the first few lines of the first file:
for line in dataset.take(5):
  print(line.numpy())

'''
-----------------------------------------------------------------------------------------------------------------
To alternate lines between files use Dataset.interleave. 
This makes it easier to shuffle files together. 
Here are the first, second and third lines from each translation:
-----------------------------------------------------------------------------------------------------------------
'''
files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
  if i % 3 == 0:
    print()
  print(line.numpy())

'''
----------------------------------------------------------------------------------------------------------------
By default, 
a TextLineDataset yields every line of each file, which may not be desirable, for example, 
if the file starts with a header line, or contains comments. 
These lines can be removed using the Dataset.skip() or Dataset.filter() transformations. 
Here we skip the first line, then filter to find only survivors.
----------------------------------------------------------------------------------------------------------------
'''
titanic_file = tf.keras.utils.get_file(
    "train.csv", 
    "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
  )

titanic_lines = tf.data.TextLineDataset(titanic_file)

for line in titanic_lines.take(10):
  print(line.numpy())

def survived(line):
  return tf.not_equal(tf.strings.substr(line, 0, 1), "0")

survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
  print(line.numpy())

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Consuming CSV data                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
The CSV file format is a popular format for storing tabular data in plain text.
--------------------------------------------------------------------------------------------------------------
'''
# For example:
titanic_file = tf.keras.utils.get_file(
    "train.csv", 
    "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
  )

df = pd.read_csv(titanic_file, index_col=None)
print(df.head())

print()

'''
-------------------------------------------------------------------------------------------------------------------
If your data fits in memory the same Dataset.from_tensor_slices method works on dictionaries, 
allowing this data to be easily imported:
-------------------------------------------------------------------------------------------------------------------
'''
titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

'''
-------------------------------------------------------------------------------------------------------------------
A more scalable approach is to load from disk as necessary.

The tf.data module provides methods to extract records from one or more CSV files that comply with RFC 4180.

The experimental.make_csv_dataset function is the high level interface for reading sets of csv files. 
It supports column type inference and many other features, like batching and shuffling, to make usage simple.
-------------------------------------------------------------------------------------------------------------------
'''
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived"
  )

for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

# You can use the select_columns argument if you only need a subset of columns.
titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", 
    select_columns=['class', 'fare', 'survived']
  )

for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

'''
--------------------------------------------------------------------------------------------------------------------
There is also a lower-level experimental.CsvDataset class which provides finer grained control. 
It does not support column type inference. Instead you must specify the type of each column.
--------------------------------------------------------------------------------------------------------------------
'''
titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
  print([item.numpy() for item in line])

# If some columns are empty, this low-level interface allows you to provide default values instead of column types.
