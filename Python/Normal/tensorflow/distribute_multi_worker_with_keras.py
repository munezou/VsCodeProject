'''
------------------------------------------------------------------------------------------
distribute
    Multi-worker training with Keras

Overview

This tutorial demonstrates multi-worker distributed training with Keras model using tf.distribute.Strategy API. 
With the help of the strategies specifically designed for multi-worker training, 
a Keras model that was designed to run on single-worker can seamlessly work on multiple workers with minimal code change.

Distributed Training in TensorFlow guide is available for an overview of the distribution strategies TensorFlow supports 
for those interested in a deeper understanding of tf.distribute.Strategy APIs.
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
        '       Preparing dataset                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Now, let's prepare the MNIST dataset from TensorFlow Datasets. 
The MNIST dataset comprises 60,000 training examples and 10,000 test examples of the handwritten digits 0–9, 
formatted as 28x28-pixel monochrome images.
---------------------------------------------------------------------------------------------------------------
'''
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(
                                name='mnist',
                                data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                                with_info=True,
                                as_supervised=True
                            )

    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

train_datasets = make_datasets_unbatched().batch(BATCH_SIZE)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Build the Keras model                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Here we use tf.keras.Sequential API to build and compile a simple convolutional neural networks Keras model 
to train with our MNIST dataset.

Note: 
For a more comprehensive walkthrough of building Keras model, please see TensorFlow Keras Guide.
---------------------------------------------------------------------------------------------------------------
'''
def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

'''
-----------------------------------------------------------------------------------------------------------------
Let's first try training the model for a small number of epochs and observe the results in single worker 
to make sure everything works correctly. 
You should expect to see the loss dropping and accuracy approaching 1.0 as epoch advances.
------------------------------------------------------------------------------------------------------------------
'''
single_worker_model = build_and_compile_cnn_model()

single_worker_model.fit(x=train_datasets, epochs=3)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Multi-worker Configuration                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Now let's enter the world of multi-worker training. 
In TensorFlow, TF_CONFIG environment variable is required for training on multiple machines, each of which possibly has a different role. 
TF_CONFIG is used to specify the cluster configuration on each worker that is part of the cluster.

There are two components of TF_CONFIG: cluster and task. 
cluster provides information about the training cluster, which is a dict consisting of different types of jobs such as worker. 
In multi-worker training, there is usually one worker that takes on a little more responsibility like saving checkpoint 
and writing summary file for TensorBoard in addition to what a regular worker does. 
Such worker is referred to as the 'chief' worker, and it is customary that the worker with index 0 is appointed as the chief worker 
(in fact this is how tf.distribute.Strategy is implemented). 
task on the other hand provides information of the current task.

In this example, we set the task type to "worker" and the task index to 0. 
This means the machine that has such setting is the first worker, which will be appointed as the chief worker and do more work than other workers. 
Note that other machines will need to have TF_CONFIG environment variable set as well, and it should have the same cluster dict, 
but different task type or task index depending on what the roles of those machines are.

For illustration purposes, this tutorial shows how one may set a TF_CONFIG with 2 workers on localhost. 
In practice, users would create multiple workers on external IP addresses/ports, and set TF_CONFIG on each worker appropriately.

Warning: 
Do not execute the following code in Colab. 
TensorFlow's runtime will attempt to create a gRPC server at the specified IP address and port, which will likely fail.

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
Note that while the learning rate is fixed in this example, in general it may be necessary to adjust the learning rate based on the global batch size.
------------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Choose the right strategy                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
In TensorFlow, 
distributed training consists of synchronous training, where the steps of training are synced across the workers and replicas, 
and asynchronous training, where the training steps are not strictly synced.

MultiWorkerMirroredStrategy, which is the recommended strategy for synchronous multi-worker training, 
will be demonstrated in this guide. To train the model, use an instance of tf.distribute.experimental.MultiWorkerMirroredStrategy. 
MultiWorkerMirroredStrategy creates copies of all variables in the model's layers on each device across all workers. 
It uses CollectiveOps, a TensorFlow op for collective communication, to aggregate gradients and keep the variables in sync. 
The tf.distribute.Strategy guide has more details about this strategy.
---------------------------------------------------------------------------------------------------------------
'''

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

'''
---------------------------------------------------------------------------------------------------------------
Note: 
TF_CONFIG is parsed and TensorFlow's GRPC servers are started at the time MultiWorkerMirroredStrategy.__init__() is called, 
so TF_CONFIG environment variable must be set before a tf.distribute.Strategy instance is created.

MultiWorkerMirroredStrategy provides multiple implementations via the CollectiveCommunication parameter. 
RING implements ring-based collectives using gRPC as the cross-host communication layer. 
NCCL uses Nvidia's NCCL to implement collectives. 
AUTO defers the choice to the runtime. 
The best choice of collective implementation depends upon the number and kind of GPUs, 
and the network interconnect in the cluster.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model with MultiWorkerMirroredStrategy                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
With the integration of tf.distribute.Strategy API into tf.keras, 
the only change you will make to distribute the training to multi-worker is enclosing the model building and model.compile() call inside strategy.scope(). 
The distribution strategy's scope dictates how and where the variables are created, and in the case of MultiWorkerMirroredStrategy, 
the variables created are MirroredVariables, and they are replicated on each of the workers.

Note: 
Currently there is a limitation in MultiWorkerMirroredStrategy where TensorFlow ops need to be created after the instance of strategy is created. 
If you see RuntimeError: 
Collective ops must be configured at program startup, try creating the instance of MultiWorkerMirroredStrategy at the beginning of the program 
and put the code that may create ops after the strategy is instantiated.

Note: 
In this Colab, the following code can run with expected result, 
but however this is effectively single-worker training since TF_CONFIG is not set. 
Once you set TF_CONFIG in your own example, you should expect speed-up with training on multiple machines.
-----------------------------------------------------------------------------------------------------------------
'''

NUM_WORKERS = 2
# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within 
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(x=train_datasets, epochs=3)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dataset sharding and batch size                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
In multi-worker training, sharding data into multiple parts is needed to ensure convergence and performance. 
However, note that in above code snippet, the datasets are directly sent to model.fit() without needing to shard; 
this is because tf.distribute.Strategy API takes care of the dataset sharding automatically in multi-worker trainings.

If you prefer manual sharding for your training, 
automatic sharding can be turned off via tf.data.experimental.DistributeOptions api. Concretely,
-------------------------------------------------------------------------------------------------------------
'''
options = tf.data.Options()
#options.experimental_distribute = tf.data.experimental.AutoShardPolicy(OFF)
train_datasets_no_auto_shard = train_datasets.with_options(options)

'''
------------------------------------------------------------------------------------------------------------
Another thing to notice is the batch size for the datasets. 
In the code snippet above, 
we use GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS, which is NUM_WORKERS times as large as the case it was for single worker, 
because the effective per worker batch size is 
the global batch size (the parameter passed in tf.data.Dataset.batch()) divided by the number of workers, 
and with this change we are keeping the per worker batch size same as before.
-------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Performance                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
You now have a Keras model that is all set up to run in multiple workers with MultiWorkerMirroredStrategy. 
You can try the following techniques to tweak performance of multi-worker training.

	* MultiWorkerMirroredStrategy provides multiple collective communication implementations. 
	RING implements ring-based collectives using gRPC as the cross-host communication layer. 
	NCCL uses Nvidia's NCCL to implement collectives. AUTO defers the choice to the runtime. 
	The best choice of collective implementation depends upon the number and kind of GPUs, and the network interconnect in the cluster. 
	To override the automatic choice, 
	specify a valid value to the communication parameter of MultiWorkerMirroredStrategy's constructor,
	e.g. communication=tf.distribute.experimental.CollectiveCommunication.NCCL.
	
	* Cast the variables to tf.float if possible. The official ResNet model includes an example of how this can be done.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Fault tolerance                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
In synchronous training, the cluster would fail if one of the workers fails and no failure-recovery mechanism exists. 
Using Keras with tf.distribute.Strategy comes with the advantage of fault tolerance in cases where workers die or are otherwise unstable. 
We do this by preserving training state in the distributed file system of your choice, 
such that upon restart of the instance that previously failed or preempted, the training state is recovered.

Since all the workers are kept in sync in terms of training epochs and steps, 
other workers would need to wait for the failed or preempted worker to restart to continue.
----------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       ModelCheckpoint callback                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To take advantage of fault tolerance in multi-worker training, 
provide an instance of tf.keras.callbacks.ModelCheckpoint at the tf.keras.Model.fit() call. 
The callback will store the checkpoint and training state in the directory corresponding to the filepath argument to ModelCheckpoint.
---------------------------------------------------------------------------------------------------------------
'''
# Replace the `filepath` argument with a path in the file system
# accessible by all workers.
#callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/keras-ckpt')]
file_path = str(PROJECT_ROOT_DIR.joinpath('tmp', 'keras-ckpt'))
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=file_path)]

with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(x=train_datasets, epochs=3, callbacks=callbacks)

data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print(
        '       finished       distribute_multi_worker_with_keras.py                      ({0})       \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()