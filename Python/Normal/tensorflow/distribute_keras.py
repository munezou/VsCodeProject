'''
------------------------------------------------------------------------------------------
distribute
    Distributed training with Keras

Overview
The tf.distribute.Strategy API provides an abstraction for distributing your training across multiple processing units. 
The goal is to allow users to enable distributed training using existing models and training code, with minimal changes.

This tutorial uses the tf.distribute.MirroredStrategy, which does in-graph replication with synchronous training on many GPUs on one machine. 
Essentially, it copies all of the model's variables to each processor. 
Then, it uses all-reduce to combine the gradients from all processors and applies the combined value to all copies of the model.

MirroredStrategy is one of several distribution strategy available in TensorFlow core. 
You can read about more strategies at distribution strategy guide.
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
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Keras API                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This example uses the tf.keras API to build the model and training loop. 
For custom training loops, see the tf.distribute.Strategy with training loops tutorial.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the dataset                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
----------------------------------------------------------------------------------------------------------------
Download the MNIST dataset and load it from TensorFlow Datasets. This returns a dataset in tf.data format.

Setting with_info to True includes the metadata for the entire dataset, 
which is being saved here to info. Among other things, this metadata object includes the number of train and test examples.
----------------------------------------------------------------------------------------------------------------
'''

datasets, info = tfds.load(name='mnist', data_dir=PROJECT_ROOT_DIR.joinpath('Data'), with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define distribution strategy                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Create a MirroredStrategy object. 
This will handle distribution, and provides a context manager (tf.distribute.MirroredStrategy.scope) to build your model inside.
---------------------------------------------------------------------------------------------------------------
'''
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup input pipeline                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size. 
In general, use the largest batch size that fits the GPU memory, and tune the learning rate accordingly.
--------------------------------------------------------------------------------------------------------------
'''
# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

'''
-------------------------------------------------------------------------------------------------------------
Pixel values, which are 0-255, have to be normalized to the 0-1 range. Define this scale in a function.
-------------------------------------------------------------------------------------------------------------
'''
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

'''
------------------------------------------------------------------------------------------------------------
Apply this function to the training and test data, shuffle the training data, and batch it for training. 
Notice we are also keeping an in-memory cache of the training data to improve performance.
------------------------------------------------------------------------------------------------------------
'''
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Create and compile the Keras model in the context of strategy.scope.
---------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
                    loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy']
                )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the callbacks                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The callbacks used here are:

	* TensorBoard: This callback writes a log for TensorBoard which allows you to visualize the graphs.
	* Model Checkpoint: This callback saves the model after every epoch.
	* Learning Rate Scheduler: Using this callback, you can schedule the learning rate to change after every epoch/batch.

For illustrative purposes, add a print callback to display the learning rate in the notebook.
---------------------------------------------------------------------------------------------------------------
'''

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = PROJECT_ROOT_DIR.joinpath('training_checkpoints')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {0} is {1}'.format(epoch + 1, model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=PROJECT_ROOT_DIR.joinpath('logs')),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train and evaluate                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Now, train the model in the usual way, 
calling fit on the model and passing in the dataset created at the beginning of the tutorial. 
This step is the same whether you are distributing the training or not.
--------------------------------------------------------------------------------------------------------------
'''
model.fit(train_dataset, epochs=12, callbacks=callbacks)

'''
--------------------------------------------------------------------------------------------------------------
As you can see below, the checkpoints are getting saved.

To see how the model perform, load the latest checkpoint and call evaluate on the test data.

Call evaluate as before using appropriate datasets.
---------------------------------------------------------------------------------------------------------------
'''

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Export to SavedModel                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Export the graph and the variables to the platform-agnostic SavedModel format. 
After your model is saved, you can load it with or without the scope.
---------------------------------------------------------------------------------------------------------------
'''
path = str(PROJECT_ROOT_DIR.joinpath('saved_model'))
model.save(path, save_format='tf')

# Load the model without strategy.scope.
unreplicated_model = tf.keras.models.load_model(path)

unreplicated_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print()
print('Eval loss: {}, Eval Accuracy: {}\n\n'.format(eval_loss, eval_acc))

'''
---------------------------------------------------------------------------------------------------------------
Load the model with strategy.scope.
---------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    replicated_model = tf.keras.models.load_model(path)
    
    tf.compat.v1.initialize_all_variables()
    
    replicated_model.compile(
                                loss='sparse_categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=['accuracy']
                            )

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print ('Eval loss: {}, Eval Accuracy: {}\n\n'.format(eval_loss, eval_acc))


data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print(
        '       finished       distribute_keras.py                               ({0})       \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()