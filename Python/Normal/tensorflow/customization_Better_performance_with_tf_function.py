'''
------------------------------------------------------------------------------------------
customization
    Better performance with tf.function

In TensorFlow 2.0, eager execution is turned on by default. 
The user interface is intuitive and flexible (running one-off operations is much easier and faster), 
but this can come at the expense of performance and deployability.

To get peak performance and to make your model deployable anywhere, 
use tf.function to make graphs out of your programs. 
Thanks to AutoGraph, a surprising amount of Python code just works with tf.function, 
but there are still pitfalls to be wary of.

The main takeaways and recommendations are:

* Don't rely on Python side effects like object mutation or list appends.
* tf.function works best with TensorFlow ops, rather than NumPy ops or Python primitives.
* When in doubt, use the for x in y idiom.
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

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(error_class))

'''
--------------------------------------------------------------------------------------------------------------
A tf.function you define is just like a core TensorFlow operation: You can execute it eagerly; 
you can use it in a graph; it has gradients; and so on.
--------------------------------------------------------------------------------------------------------------
'''
# A function is like an op

@tf.function
def add(a, b):
    return a + b

func_add = add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
print('func_add = \n{0}\n'.format(func_add))

# Functions have gradients

@tf.function
def add(a, b):
    return a + b

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)

tape_gradient = tape.gradient(result, v)
print('tape_gradient = \n{0}\n'.format(tape_gradient))

# You can use functions inside functions

@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

ds_layer = dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
print('ds_layer = \n{0}\n'.format(ds_layer))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Tracing and polymorphism                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Python's dynamic typing means that you can call functions with a variety of argument types, 
and Python will do something different in each scenario.

On the other hand, TensorFlow graphs require static dtypes and shape dimensions. 
tf.function bridges this gap by retracing the function when necessary to generate the correct graphs. 
Most of the subtlety of tf.function usage stems from this retracing behavior.

You can call a function with arguments of different types to see what is happening.
--------------------------------------------------------------------------------------------------------------
'''
# Functions are polymorphic

@tf.function
def double(a):
    print("Tracing with", a)
    return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

'''
-------------------------------------------------------------------------------------------------------------
To control the tracing behavior, use the following techniques:

* Create a new tf.function. Separate tf.function objects are guaranteed not to share traces.
* Use the get_concrete_function method to get a specific trace
* Specify input_signature when calling tf.function to trace only once per calling graph.
-------------------------------------------------------------------------------------------------------------
'''

print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")

with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       When to retrace?                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A polymorphic tf.function keeps a cache of concrete functions generated by tracing. 
The cache keys are effectively tuples of keys generated from the function args and kwargs. 
The key generated for a tf.Tensor argument is its shape and type. 
The key generated for a Python primitive is its value. 
For all other Python types, 
the keys are based on the object id() so that methods are traced independently for each instance of a class. 
In the future, 
TensorFlow may add more sophisticated caching for Python objects that can be safely converted to tensors.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Python or Tensor args?                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Often, Python arguments are used to control hyperparameters and graph constructions - for example, 
num_layers=10 or training=True or nonlinearity='relu'. 
So if the Python argument changes, it makes sense that you'd have to retrace the graph.

However, it's possible that a Python argument is not being used to control graph construction. 
In these cases, a change in the Python value can trigger needless retracing. 
Take, for example, this training loop, which AutoGraph will dynamically unroll. 
Despite the multiple traces, the generated graph is actually identical, so this is a bit inefficient.
---------------------------------------------------------------------------------------------------------------
'''
def train_one_step():
    pass

@tf.function
def train(num_steps):
    print("Tracing with num_steps = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()

train(num_steps=10)
train(num_steps=20)

'''
---------------------------------------------------------------------------------------------------------------
The simple workaround here is to cast your arguments to Tensors if they do not affect the shape of the generated graph.
---------------------------------------------------------------------------------------------------------------
'''
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Side effects in tf.function                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
In general, Python side effects (like printing or mutating objects) only happen during tracing. 
So how can you reliably trigger side effects from tf.function?

The general rule of thumb is to only use Python side effects to debug your traces. Otherwise, 
TensorFlow ops like tf.Variable.assign, tf.print, 
and tf.summary are the best way to ensure your code will be traced and executed by the TensorFlow runtime with each call. 
In general using a functional style will yield the best results.
----------------------------------------------------------------------------------------------------------------
'''
