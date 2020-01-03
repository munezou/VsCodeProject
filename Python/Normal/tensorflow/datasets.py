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
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
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
def process_categorical_data(data, categories):
    """
    ------------------------------------------------------------------
    Returns a one-hot encoded tensor representing category values.
    ------------------------------------------------------------------
    """
    
    # Remove the first ' '
    data = tf.strings.regex_replace(data, '^ ', '')
    # Remove the last '.'
    data = tf.strings.regex_replace(data, r'\.$', '')
    
    # One-hot encoding
    # Reshape data from one dimension (list) to two dimensions (list of one element list).
    data = tf.reshape(data, [-1, 1])
    
    # For each element, create a list of the boolean values of the number of categories, 
    # where the label of the element and the category match are True.
    data = categories == data
    
    # Casts a boolean to a floating point number.
    data = tf.cast(data, tf.float32)
    
    # You can also put the entire encoding on one line:
    # data = tf.cast(categories == tf.reshape(data, [-1, 1]), tf.float32)
    return data

'''
-----------------------------------------------------------------------
To visualize this process, 
we take one tensor of the category column from the first batch, process it, 
and show the state before and after.
------------------------------------------------------------------------
'''
class_tensor = examples['class']
print('class_tensor = \n{0}\n'.format(class_tensor))

class_categories = CATEGORIES['class']
print('class_categories = {0}\n'.format(class_categories))

processed_class = process_categorical_data(class_tensor, class_categories)
print('processed_class = \n{0}\n'.format(processed_class))

'''
----------------------------------------------------------------------------
Notice the relationship between the length of the two inputs and the shape of the output.
----------------------------------------------------------------------------
'''
print('Size of batch: {0}'.format(len(class_tensor.numpy())))
print('Number of category labels: {0}'.format(len(class_categories)))
print('Shape of one-hot encoded tensor: {0}\n'.format(processed_class.shape))

'''
----------------------------------------------------------------------------
Continuous data
Continuous data must be normalized so that the value is between 0 and 1. 
To do this, 
write a function that multiplies each value by 1 divided by twice the average of the column values.
----------------------------------------------------------------------------
'''

# This function also reshapes the data into a two-dimensional tensor.
def process_continuous_data(data, mean):
    # standardization of data
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])

'''
---------------------------------------------------------------------------
To perform this calculation, you need the average of the column values. 
Obviously, in reality it is necessary to calculate this value, 
but we will show a value for this example.
---------------------------------------------------------------------------
'''
MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}


'''
-----------------------------------------------------------------------------
To perform this calculation, you need the average of the column values. 
Obviously, in reality it is necessary to calculate this value, 
but we will show a value for this example.
-----------------------------------------------------------------------------
'''
# To see what this function actually does, 
# take a continuous tensor and look before and after.
age_tensor = examples['age']
print('age_tensor = \n{0}\n'.format(age_tensor))

age_tensor_standard = process_continuous_data(age_tensor, MEANS['age'])
print('age_tensor_standard = \n{0}\n'.format(age_tensor_standard))

'''
------------------------------------------------------------------------------
Data preprocessing

Combine these preprocessing tasks into a single function 
that can be mapped to batches in a dataset.
------------------------------------------------------------------------------
'''
def preprocess(features, labels):
        
    # Processing category features
    for feature in CATEGORIES.keys():
        features[feature] = process_categorical_data(features[feature], CATEGORIES[feature])

    # Processing of continuous features
    for feature in MEANS.keys():
        features[feature] = process_continuous_data(features[feature], MEANS[feature])
    
    # Assemble features into one tensor.
    features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)
    
    return features, labels

# Then apply it using the tf.Dataset.map function and shuffle the dataset to prevent overtraining.
train_data = raw_train_data.map(preprocess).shuffle(500)
test_data = raw_test_data.map(preprocess)

# Let's see what one sample looks like.
examples, labels = next(iter(train_data))

print('examples = \n{0}\n'.format(examples))
print('labels = \n{0}\n'.format(labels))

'''
----------------------------------------------------------------------------------
This example consists of a two-dimensional array with 12 items (the batch size). 
Each item represents one line of the original CSV file. 
Labels are one-dimensional tensors with 12 values.
-----------------------------------------------------------------------------------
'''

'''
-----------------------------------------------------------------------------------
Build the model

In this example, 
we use the Keras Functional API and wrap it with the get_model constructor to build a simple model.
-----------------------------------------------------------------------------------
'''

def get_model(input_dim, hidden_units=[100]):
    """
    -----------------------------------------------------
    Create Keras model with multiple layers

    argument:
        input_dim: (int) shape of items in batch
        labels_dim: (int) label shape
        hidden_units: [int] Layer size of DNN (input layer first)
        learning_rate: (float) optimizer learning rate

    Return value:
        Keras model
    ------------------------------------------------------
    """

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

'''
--------------------------------------------------------------------
The get_model constructor needs 
to know the shape of the input data (except the batch size).
--------------------------------------------------------------------
'''
train_element_spec = train_data.element_spec

input_shape = train_element_spec[0]
output_shape = train_element_spec[1] # [0] is the batch size

input_dimension = input_shape.shape.dims[1]

print('(input_shape = {0}, output_shape = {1})\n'.format(input_shape, output_shape))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Training, evaluation, and prediction                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
model = get_model(input_dimension)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_data, epochs=20)

# Once you have trained your model, you can check the accuracy rate on the test_data dataset.
test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {0}, Test Accuracy {1}\n'.format(test_loss, test_accuracy))

# Use tf.keras.Model.predict to infer the labels of a single batch or a dataset consisting of batches.
predictions = model.predict(test_data)

# View some of the results.
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
    " | Actual outcome: ",
    ("SURVIVED" if bool(survived) else "DIED"))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Loading from .npz files                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file(
        PROJECT_ROOT_DIR.joinpath('original_data/datasets/Mnist/mnist.npz'),
        DATA_URL
    )

with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Load a NumPy array using tf.data.Dataset.                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Suppose you have an array of samples and an array of corresponding labels. 
Input these two arrays to tf.data.Dataset.from_tensor_slices as tuples to create tf.data.Dataset.
---------------------------------------------------------------------------------------------------------------
'''
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

train_element_spec = train_dataset.element_spec
print('train_element_spec = \n{0}\n'.format(train_element_spec))

test_element_spec = test_dataset.element_spec
print('test_element_spec = \n{0}\n'.format(test_element_spec))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Using datasets                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Shuffle and batch datasets
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '               Model building and training                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile   (
                    optimizer=tf.keras.optimizers.RMSprop(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
                )

print('---< fitting model >---')
model_fit = model.fit(train_dataset, epochs=10)

print('---< Evaluate the model. >---')
model.evaluate(test_dataset)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load a pandas DataFrame using tf.data.                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
contents)

This tutorial shows an example of loading a pandas DataFrame and loading the data into tf.data.Dataset.

This tutorial uses a small dataset provided by the Cleveland Clinic Foundation for Heart Disease. 
This dataset (CSV) contains hundreds of rows of data. 
The rows represent each patient and the columns represent various attributes.

This data can be used to determine and predict whether a patient has heart disease. 
This is a binary classification problem.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Read data using pandas.                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Download the CSV containing the heart dataset.
DATA_URL = 'https://storage.googleapis.com/applied-dl/heart.csv'

csv_file = tf.keras.utils.get_file(
        PROJECT_ROOT_DIR.joinpath('original_data/datasets/Heart/heart.csv'),
        DATA_URL
    )

# Read the csv using pandas
df = pd.read_csv(csv_file)

print('df.head() = \n{0}\n'.format(df.head()))

print('df.dtypes = \n{0}\n'.format(df.dtypes))

# Converts the only object type thal column in a dataframe to a discrete value.
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print('df.head() = \n{0}\n'.format(df.head()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load data using tf.data.Dataset.                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# Display the first 5 lines of tf.data.Dataset.
for feat, targ in dataset.take(5):
    print ('Features: {0}, Target: {1}'.format(feat, targ))

'''
-------------------------------------------------------------------------------------------------------------
pd.Series implements the __array__ protocol, so it can be used almost anywhere you use np.array or tf.Tensor.
-------------------------------------------------------------------------------------------------------------
'''
tf_constant_df = tf.constant(df['thal'])
print('tf.constant(df["thal"]) = \n{0}\n'.format(tf_constant_df))

# Performs batch processing by shuffling data.
train_dataset = dataset.shuffle(len(df)).batch(1)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create and train a model.                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# create model
model = get_compiled_model()

print('---< fitting model >---')
model.fit(train_dataset, epochs=15)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Alternate feature columns                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Passing dictionary data to the input to the model is as easy as creating a dictionary of the same type in tf.keras.layers.Input, 
applying some preprocessing, and stacking using functional api. 
Can be done. 
This can be used instead of the feature column.
---------------------------------------------------------------------------------------------------------------
'''
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

'''
-------------------------------------------------------------------------------------------------------------------
The easiest way to preserve the column structure of a pandas DataFrame 
when using tf.data is to convert the DataFrame to dictionary data and truncate it.
-------------------------------------------------------------------------------------------------------------------
'''
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print ('dict_slice = \n{0}\n'.format(dict_slice))

model_func.fit(dict_slices, epochs=15)

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
cash_path = str(PROJECT_ROOT_DIR.joinpath("cache.tf-data"))

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