'''
------------------------------------------------------------------------------------------
distribute
    Multi-worker training with Estimator

Overview

Note: 
While you can use Estimators with tf.distribute API, 
we would instead recommend you use Keras with tf.distribute (please see Multi-worker Training with Keras). 
Estimator training with tf.distribute.Strategy has limited support at this time.

This tutorial demonstrates how tf.distribute.Strategy can be used for distributed multi-worker training with tf.estimator. 
If you write your code using tf.estimator, and you're interested in scaling beyond a single machine with high performance, 
this tutorial is for you.

Before getting started, please read the tf.distribute.Strategy guide. 
The multi-GPU training tutorial is also relevant, because this tutorial uses the same model.
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

print(__doc__)

tfds.disable_progress_bar()

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Input function                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This tutorial uses the MNIST dataset from TensorFlow Datasets. 
The code here is similar to the multi-GPU training tutorial with one key difference: 
when using Estimator for multi-worker training, it is necessary to shard the dataset by the number of workers to ensure model convergence. 
The input data is sharded by worker index, so that each worker processes 1/num_workers distinct portions of the dataset.
---------------------------------------------------------------------------------------------------------------
'''
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
    datasets, info = tfds.load(
                                name='mnist',
                                data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
                                with_info=True,
                                as_supervised=True
                            )
    
    mnist_dataset = (
                        datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test']
                    )

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    if input_context:
        mnist_dataset = mnist_dataset.shard(
                                            input_context.num_input_pipelines,
                                            input_context.input_pipeline_id
                                            )
        
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

'''
------------------------------------------------------------------------------------------------------
Another reasonable approach to achieve convergence 
would be to shuffle the dataset with distinct seeds at each worker.
------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Multi-worker configuration                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------------
One of the key differences in this tutorial (compared to the multi-GPU training tutorial) is the multi-worker setup. 
The TF_CONFIG environment variable is the standard way to specify the cluster configuration to each worker that is part of the cluster.

There are two components of TF_CONFIG: cluster and task. 
cluster provides information about the entire cluster, namely the workers and parameter servers in the cluster. 
task provides information about the current task. 
In this example, the task type is worker and the task index is 0.

For illustration purposes, 
this tutorial shows how to set a TF_CONFIG with 2 workers on localhost. 
In practice, you would create multiple workers on an external IP address and port, 
and set TF_CONFIG on each worker appropriately, i.e. modify the task index.

Warning: 
Do not execute the following code in Colab. 
TensorFlow's runtime will attempt to create a gRPC server at the specified IP address and port, which will likely fail.

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
					'worker': ["localhost:12345", "localhost:23456"]
				},
    'task': {'type': 'worker', 'index': 0}
})
---------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------
Write the layers, the optimizer, and the loss function for training. 
This tutorial defines the model with Keras layers, similar to the multi-GPU training tutorial.
---------------------------------------------------------------------------------------------------------
'''
LEARNING_RATE = 1e-4

def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=LEARNING_RATE
        )
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )(labels, logits)
    
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(
                        loss, tf.compat.v1.train.get_or_create_global_step()
                    )
        )

'''
-------------------------------------------------------------------------------------------
Note: 
Although the learning rate is fixed in this example, 
in general it may be necessary to adjust the learning rate based on the global batch size.
-------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       MultiWorkerMirroredStrategy                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
To train the model, use an instance of tf.distribute.experimental.MultiWorkerMirroredStrategy. 
MultiWorkerMirroredStrategy creates copies of all variables in the model's layers on each device across all workers. 
It uses CollectiveOps, a TensorFlow op for collective communication, to aggregate gradients and keep the variables in sync. 
The tf.distribute.Strategy guide has more details about this strategy.
------------------------------------------------------------------------------------------
'''
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
