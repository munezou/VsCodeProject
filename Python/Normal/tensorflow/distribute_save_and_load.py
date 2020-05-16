'''
------------------------------------------------------------------------------------------
distribute
    Save and load a model using a distribution strategy

Overview

It's common to save and load a model during training. 
There are two sets of APIs for saving and loading a keras model: a high-level API, and a low-level API. 
This tutorial demonstrates how you can use the SavedModel APIs when using tf.distribute.Strategy. 
To learn about SavedModel and serialization in general, 
please read the saved model guide, and the Keras model serialization guide. Let's start with a simple example:
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import json
import contextlib
from pathlib import Path
from packaging import version
from PIL import Image
import tempfile

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub

print(__doc__)

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

# Prepare the data and model using tf.distribute.Strategy:
mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
    datasets, ds_info = tfds.load(
                                    name='mnist', 
                                    data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                                    with_info=True, 
                                    as_supervised=True
                                )
    
    mnist_train, mnist_test = datasets['train'], datasets['test']

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    return train_dataset, eval_dataset

def get_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])
        return model

# Train the model:
model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs=2)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Save and load the model                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Now that you have a simple model to work with, let's take a look at the saving/loading APIs. 
There are two sets of APIs available:

	* High level keras model.save and tf.keras.models.load_model
	* Low level tf.saved_model.save and tf.saved_model.load
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The Keras APIs                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Here is an example of saving and loading a model with the Keras APIs:
---------------------------------------------------------------------------------------------------------------
'''
keras_model_path = str(PROJECT_ROOT_DIR.joinpath('tmp', 'keras_save'))
model.save(keras_model_path)  # save() should be called out of strategy scope

'''
---------------------------------------------------------------------------------------------------------------
Restore the model without tf.distribute.Strategy:
---------------------------------------------------------------------------------------------------------------
'''
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)

'''
----------------------------------------------------------------------------------------------------------------
After restoring the model, you can continue training on it, 
even without needing to call compile() again, since it is already compiled before saving. 
The model is saved in the TensorFlow's standard SavedModel proto format. For more information, 
please refer to the guide to saved_model format.

It is important to only call the model.save() method out of the scope of tf.distribute.strategy. 
Calling it within the scope is not supported.

Now to load the model and train it using a tf.distribute.Strategy:
----------------------------------------------------------------------------------------------------------------
'''
another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

with another_strategy.scope():
    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
    restored_keras_model_ds.fit(train_dataset, epochs=2)

'''
---------------------------------------------------------------------------------------------------------------
As you can see, loading works as expected with tf.distribute.Strategy. 
The strategy used here does not have to be the same strategy used before saving.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The tf.saved_model APIs                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
Now let's take a look at the lower level APIs. 
Saving the model is similar to the keras API:
----------------------------------------------------------------------------------------------------------------
'''
model = get_model()  # get a fresh model
saved_model_path = str(PROJECT_ROOT_DIR.joinpath('tmp', 'tf_save'))
tf.saved_model.save(model, saved_model_path)

'''
----------------------------------------------------------------------------------------------------------------
Loading can be done with tf.saved_model.load(). However, 
since it is an API that is on the lower level (and hence has a wider range of use cases), 
it does not return a Keras model. 
Instead, 
it returns an object that contain functions that can be used to do inference. For example:
----------------------------------------------------------------------------------------------------------------
'''
DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

'''
----------------------------------------------------------------------------------------------------------------
The loaded object may contain multiple functions, each associated with a key. 
The "serving_default" is the default key for the inference function with a saved Keras model. 
To do an inference with this function:
----------------------------------------------------------------------------------------------------------------
'''
predict_dataset = eval_dataset.map(lambda image, label: image)

for batch in predict_dataset.take(1):
    print(inference_func(batch))

'''
----------------------------------------------------------------------------------------------------------------
You can also load and do inference in a distributed manner:
----------------------------------------------------------------------------------------------------------------
'''
another_strategy = tf.distribute.MirroredStrategy()

with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

    dist_predict_dataset = another_strategy.experimental_distribute_dataset(
                                predict_dataset
                            )

    # Calling the function in a distributed manner
    for batch in dist_predict_dataset:
        another_strategy.experimental_run_v2(inference_func, args=(batch,))

'''
---------------------------------------------------------------------------------------------------------------
Calling the restored function is just a forward pass on the saved model (predict). 
What if yout want to continue training the loaded function? 
Or embed the loaded function into a bigger model? 
A common practice is to wrap this loaded object to a Keras layer to achieve this. 
Luckily, TF Hub has hub.KerasLayer for this purpose, shown here:
--------------------------------------------------------------------------------------------------------------
'''
def build_model(loaded):
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    # Wrap what's loaded to a KerasLayer
    keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model

another_strategy = tf.distribute.MirroredStrategy()

with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    model = build_model(loaded)

    model.compile(
                    loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy']
                )
    
    model.fit(train_dataset, epochs=2)

'''
---------------------------------------------------------------------------------------------------------------
As you can see, 
hub.KerasLayer wraps the result loaded back from tf.saved_model.load() into a Keras layer 
that can be used to build another model. 
This is very useful for transfer learning.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Which API should I use?                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
For saving, 
if you are working with a keras model, it is almost always recommended to use the Keras's model.save() API. 
If what you are saving is not a Keras model, then the lower level API is your only choice.

For loading, 
which API you use depends on what you want to get from the loading API. 
If you cannot (or do not want to) get a Keras model, then use tf.saved_model.load(). 
Otherwise, use tf.keras.models.load_model(). 
Note that you can get a Keras model back only if you saved a Keras model.

It is possible to mix and match the APIs. 
You can save a Keras model with model.save, 
and load a non-Keras model with the low-level API, tf.saved_model.load.
---------------------------------------------------------------------------------------------------------------
'''
model = get_model()

# Saving the model using Keras's save() API
model.save(keras_model_path) 

another_strategy = tf.distribute.MirroredStrategy()

# Loading the model using lower level API
with another_strategy.scope():
    loaded = tf.saved_model.load(keras_model_path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Caveats                                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------------------
A special case is when you have a Keras model that does not have well-defined inputs. 
For example, 
a Sequential model can be created without any input shapes (Sequential([Dense(3), ...]). 
Subclassed models also do not have well-defined inputs after initialization. 
In this case, 
you should stick with the lower level APIs on both saving and loading, otherwise you will get an error.

To check if your model has well-defined inputs, just check if model.inputs is None. 
If it is not None, 
you are all good. Input shapes are automatically defined when the model is used in .fit, 
.evaluate, .predict, or when calling the model (model(inputs)).

Here is an example:
------------------------------------------------------------------------------------------------------------
'''
class SubclassedModel(tf.keras.Model):
    
    output_name = 'output_layer'

    def __init__(self):
        super(SubclassedModel, self).__init__()
        self._dense_layer = tf.keras.layers.Dense(
                                5, 
                                dtype=tf.dtypes.float32, 
                                name=self.output_name
                            )

    def call(self, inputs):
        return self._dense_layer(inputs)

my_model = SubclassedModel()

# my_model.save(keras_model_path)  # ERROR! 
tf.saved_model.save(my_model, saved_model_path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        distribute_save_and_load.py                        (2020/05/16)                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()