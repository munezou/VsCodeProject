from __future__ import absolute_import, division, print_function, unicode_literals

'''
------------------------------------------------------------------------------------------
Training checkpoints

The phrase "Saving a TensorFlow model" typically means one of two things:

1.Checkpoints, OR
2.SavedModel.

Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model. 
Checkpoints do not contain any description of the computation defined by the model 
and thus are typically only useful when source code that will use the saved parameter values is available.

The SavedModel format on the other hand includes a serialized description of the computation defined by the model 
in addition to the parameter values (checkpoint). 
Models in this format are independent of the source code that created the model. 
They are thus suitable for deployment via TensorFlow Serving, TensorFlow Lite, TensorFlow.js, 
or programs in other programming languages (the C, C++, Java, Go, Rust, C# etc. TensorFlow APIs).

This guide covers APIs for writing and reading checkpoints.
------------------------------------------------------------------------------------------
'''
print(__doc__)

# common library
import os
import platform
import shutil
import subprocess
from packaging import version
import tensorflow as tf


print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

class Net(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)

net = Net()
print('net = \n{0}\n'.format(net))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Saving from tf.keras training APIs                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# tf.keras.Model.save_weights saves a TensorFlow checkpoint.
net.save_weights('easy_checkpoint')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Writing checkpoints                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The persistent state of a TensorFlow model is stored in tf.Variable objects. 
These can be constructed directly, 
but are often created through high-level APIs like tf.keras.layers or tf.keras.Model.

The easiest way to manage variables is by attaching them to Python objects, then referencing those objects.

Subclasses of tf.train.Checkpoint, tf.keras.layers.Layer, 
and tf.keras.Model automatically track variables assigned to their attributes. 
The following example constructs a simple linear model, 
then writes checkpoints which contain values for all of the model's variables.

You can easily save a model-checkpoint with Model.save_weights
--------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Manual checkpointing                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Setup
# To help demonstrate all the features of tf.train.Checkpoint define a toy dataset and optimization step:
def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)

def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the checkpoint objects                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To manually make a checkpoint you will need a tf.train.Checkpoint object. 
Where the objects you want to checkpoint are set as attributes on the object.

A tf.train.CheckpointManager can also be helpful for managing multiple checkpoints.
---------------------------------------------------------------------------------------------------------------
'''
opt = tf.keras.optimizers.Adam(0.1)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
#path_tf_ckpts = os.path.join(PROJECT_ROOT_DIR, "tf_ckpts")
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train and checkpoint the model                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The following training loop creates an instance of the model and of an optimizer, 
then gathers them into a tf.train.Checkpoint object. 
It calls the training step in a loop on each batch of data, and periodically writes checkpoints to disk.
---------------------------------------------------------------------------------------------------------------
'''
def train_and_checkpoint(net, manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for example in toy_dataset():
        loss = train_step(net, example, opt)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
        print("Saved checkpoint for step {0}: {1}".format(int(ckpt.step), save_path))
        print("loss {:1.2f}".format(loss.numpy()))

train_and_checkpoint(net, manager)