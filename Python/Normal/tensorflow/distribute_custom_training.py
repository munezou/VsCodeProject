'''
------------------------------------------------------------------------------------------
distribute
    Custom training with tf.distribute.Strategy

This tutorial demonstrates how to use tf.distribute.Strategy with custom training loops. 
We will train a simple CNN model on the fashion MNIST dataset. 
The fashion MNIST dataset contains 60000 train images of size 28 x 28 and 10000 test images of size 28 x 28.

We are using custom training loops to train our model because they give us flexibility and a greater control on training. 
Moreover, it is easier to debug the model and the training loop.
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
        '       Download the fashion MNIST dataset                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create a strategy to distribute the variables and the graph                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
How does tf.distribute.MirroredStrategy strategy work?

	* All the variables and the model graph is replicated on the replicas.
	* Input is evenly distributed across the replicas.
	* Each replica calculates the loss and gradients for the input it received.
	* The gradients are synced across all the replicas by summing them.
	* After the sync, the same update is made to the copies of the variables on each replica.
	
Note: You can put all the code below inside a single scope. 
We are dividing it into several code cells for illustration purposes.
----------------------------------------------------------------------------------------------------------------
'''
# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {0}\n'.format(strategy.num_replicas_in_sync))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup input pipeline                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Export the graph and the variables to the platform-agnostic SavedModel format. 
After your model is saved, you can load it with or without the scope.
---------------------------------------------------------------------------------------------------------------
'''
BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

# Create the datasets and distribute them:
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Create a model using tf.keras.Sequential. 
You can also use the Model Subclassing API to do this.
--------------------------------------------------------------------------------------------------------------
'''
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = PROJECT_ROOT_DIR.joinpath('training_checkpoints')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the loss function                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Normally, on a single machine with 1 GPU/CPU, loss is divided by the number of examples in the batch of input.
So, how should the loss be calculated when using a tf.distribute.Strategy?

	* For an example, let's say you have 4 GPU's and a batch size of 64. 
	One batch of input is distributed across the replicas (4 GPUs), each replica getting an input of size 16.

	* The model on each replica does a forward pass with its respective input and calculates the loss. 
	Now, instead of dividing the loss by the number of examples in its respective input (BATCH_SIZE_PER_REPLICA = 16), 
	the loss should be divided by the GLOBAL_BATCH_SIZE (64).

Why do this?

	* This needs to be done because after the gradients are calculated on each replica, 
	they are synced across the replicas by summing them.
	
How to do this in TensorFlow?

	* If you're writing a custom training loop, as in this tutorial, 
	you should sum the per example losses and divide the sum by the GLOBAL_BATCH_SIZE: 
	scale_loss = tf.reduce_sum(loss) * (1. / GLOBAL_BATCH_SIZE) or you can use tf.nn.compute_average_loss which takes the per example loss, 
	optional sample weights, and GLOBAL_BATCH_SIZE as arguments and returns the scaled loss.

	* If you are using regularization losses in your model then you need to scale the loss value by number of replicas. 
	You can do this by using the tf.nn.scale_regularization_loss function.

	* Using tf.reduce_mean is not recommended. 
	Doing so divides the loss by actual per replica batch size which may vary step to step.

	* This reduction and scaling is done automatically in keras model.compile and model.fit

	* If using tf.keras.losses classes (as in the example below), the loss reduction needs to be explicitly specified to be one of NONE or SUM. 
	AUTO and SUM_OVER_BATCH_SIZE are disallowed when used with tf.distribute.Strategy. 
	AUTO is disallowed because the user should explicitly think about what reduction they want to make sure it is correct in the distributed case. 
	SUM_OVER_BATCH_SIZE is disallowed because currently it would only divide by per replica batch size, 
	and leave the dividing by number of replicas to the user, which might be easy to miss. 
	So instead we ask the user do the reduction themselves explicitly.
-----------------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                        reduction=tf.keras.losses.Reduction.NONE
                    )
    
    # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the metrics to track loss and accuracy                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
----------------------------------------------------------------------------------------------------------------
These metrics track the test loss and training and test accuracy. 
You can use .result() to get the accumulated statistics at any time.
----------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Training loop                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss 

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = (
                        "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}"
                    )
        
        print (template.format(
                            epoch+1, train_loss, train_accuracy.result()*100, test_loss.result(),
                            test_accuracy.result()*100
                            )
                )

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

'''
------------------------------------------------------------------------------------------------------------------
Things to note in the example above:

	* We are iterating over the train_dist_dataset and test_dist_dataset using a for x in ... construct.
	
	* The scaled loss is the return value of the distributed_train_step. 
	This value is aggregated across replicas using the tf.distribute.Strategy.reduce call 
	and then across batches by summing the return value of the tf.distribute.Strategy.reduce calls.
	
	* tf.keras.Metrics should be updated inside train_step and test_step that gets executed by tf.distribute.Strategy.experimental_run_v2.
	
	* tf.distribute.Strategy.experimental_run_v2 returns results from each local replica in the strategy, 
	and there are multiple ways to consume this result. 
	You can do tf.distribute.Strategy.reduce to get an aggregated value. 
	You can also do tf.distribute.Strategy.experimental_local_results to get the list of values contained in the result, 
	one per local replica.
--------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Restore the latest checkpoint and test                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A model checkpointed with a tf.distribute.Strategy can be restored with or without a strategy.
---------------------------------------------------------------------------------------------------------------
'''
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

@tf.function
def eval_step(images, labels):
    predictions = new_model(images, training=False)
    eval_accuracy(labels, predictions)

checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
    eval_step(images, labels)

print ('Accuracy after restoring the saved model without strategy: {}'.format(
                eval_accuracy.result()*100
            )
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Alternate ways of iterating over a dataset                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Using iterators

If you want to iterate over a given number of steps and not through the entire dataset 
you can create an iterator using the iter call and explicity call next on the iterator. 
You can choose to iterate over the dataset both inside and outside the tf.function. 
Here is a small snippet demonstrating iteration of the dataset outside the tf.function using an iterator.
---------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    for _ in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        train_iter = iter(train_dist_dataset)

        for _ in range(10):
            total_loss += distributed_train_step(next(train_iter))
            num_batches += 1
        
        average_train_loss = total_loss / num_batches

        template = ("Epoch {}, Loss: {}, Accuracy: {}")
        print (template.format(epoch+1, average_train_loss, train_accuracy.result()*100))
        train_accuracy.reset_states()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Iterating inside a tf.function                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
You can also iterate over the entire input train_dist_dataset inside a tf.function using the for x in ... construct or by creating iterators like we did above. 
The example below demonstrates wrapping one epoch of training in a tf.function and iterating over train_dist_dataset inside the function.
--------------------------------------------------------------------------------------------------------------
'''
with strategy.scope():
    @tf.function
    def distributed_train_epoch(dataset):
        total_loss = 0.0
        num_batches = 0
        for x in dataset:
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                                args=(x,))
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            num_batches += 1
        return total_loss / tf.cast(num_batches, dtype=tf.float32)

    for epoch in range(EPOCHS):
        train_loss = distributed_train_epoch(train_dist_dataset)

        template = ("Epoch {}, Loss: {}, Accuracy: {}")
        print (template.format(epoch+1, train_loss, train_accuracy.result()*100))

        train_accuracy.reset_states()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Tracking training loss across replicas                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Note: As a general rule, you should use tf.keras.Metrics to track per-sample values 
and avoid values that have been aggregated within a replica.

We do not recommend using tf.metrics.Mean to track the training loss across different replicas, 
because of the loss scaling computation that is carried out.

For example, if you run a training job with the following characteristics:

	* Two replicas
	* Two samples are processed on each replica
	* Resulting loss values: [2, 3] and [4, 5] on each replica
	* Global batch size = 4

With loss scaling, you calculate the per-sample value of loss on each replica by adding the loss values, 
and then dividing by the global batch size. 
In this case: (2 + 3) / 4 = 1.25 and (4 + 5) / 4 = 2.25.

If you use tf.metrics.Mean to track loss across the two replicas, the result is different. 
In this example, you end up with a total of 3.50 and count of 2, 
which results in total/count = 1.75 when result() is called on the metric. 
Loss calculated with tf.keras.Metrics is scaled by an additional factor that is equal to the number of replicas in sync.
---------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        distribute_custom_training.py                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()