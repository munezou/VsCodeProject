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
import datetime

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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
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
@tf.function
def f(x):
    print("Traced with", x)
    tf.print("Executed with", x)

f(1)
f(1)
f(2)

print()

'''
----------------------------------------------------------------------------------------------------------------
If you would like to execute Python code during each invocation of a tf.function, tf.py_function is an exit hatch. 
The drawback of tf.py_function is that it's not portable or particularly performant, 
nor does it work well in distributed (multi-GPU, TPU) setups. 
Also, since tf.py_function has to be wired into the graph, it casts all inputs/outputs to tensors.
-----------------------------------------------------------------------------------------------------------------
'''

external_list = []

def side_effect(x):
    print('Python side effect')
    external_list.append(x)

@tf.function
def f1(x):
    tf.py_function(side_effect, inp=[x], Tout=[])

f1(1)
f1(1)
f1(1)

print()

assert len(external_list) == 3
# .numpy() call required because py_function casts 1 to tf.constant(1)
assert external_list[0].numpy() == 1

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Beware of Python state                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Many Python features, such as generators and iterators, rely on the Python runtime to keep track of state. 
In general, while these constructs work as expected in Eager mode, 
many unexpected things can happen inside a tf.function due to tracing behavior.

To give one example, advancing iterator state is a Python side effect and therefore only happens during tracing.
---------------------------------------------------------------------------------------------------------------
'''

external_var = tf.Variable(0)
@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print("Value of external_var:", external_var)

iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)
print()

'''
----------------------------------------------------------------------------------------------------------------
If an iterator is generated and consumed entirely within the tf.function, then it should work correctly. 
However, the entire iterator is probably being traced, which can lead to a giant graph. 
This may be what you want. 
But if you're training on an large in-memory dataset represented as a Python list, 
then this can generate a very large graph, and tf.function is unlikely to yield a speedup.

If you want to iterate over Python data, 
the safest way is to wrap it in a tf.data.Dataset and use the for x in y idiom. 
AutoGraph has special support for safely converting for loops when y is a tensor or tf.data.Dataset.
----------------------------------------------------------------------------------------------------------------
'''
def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train2(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
    return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train2, small_data)
measure_graph_size(train2, big_data)

measure_graph_size(
                    train2, 
                    tf.data.Dataset.from_generator(
                        lambda: small_data, (tf.int32, tf.int32)
                    )
                )

measure_graph_size(
                    train2, 
                    tf.data.Dataset.from_generator(
                        lambda: big_data, (tf.int32, tf.int32)
                    )
                )

'''
---------------------------------------------------------------------------------------
When wrapping Python/Numpy data in a Dataset, be mindful of tf.data.Dataset.from_generator versus tf.data.Dataset.from_tensors. 
The former will keep the data in Python and fetch it via tf.py_function which can have performance implications, 
whereas the latter will bundle a copy of the data as one large tf.constant() node in the graph, which can have memory implications.

Reading data from files via TFRecordDataset/CsvDataset/etc. is the most effective way to consume data, 
as then TensorFlow itself can manage the asynchronous loading and prefetching of data, without having to involve Python.
---------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Automatic Control Dependencies                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
A very appealing property of functions as the programming model, over a general dataflow graph, 
is that functions can give the runtime more information about what was the intended behavior of the code.

For example, 
when writing code which has multiple reads and writes to the same variables, 
a dataflow graph might not naturally encode the originally intended order of operations. In tf.function, 
we resolve ambiguities in execution order by referring to the execution order of statements in the original Python code. 
This way, ordering of stateful operations in a tf.function replicates the semantics of Eager mode.

This means there's no need to add manual control dependencies; 
tf.function is smart enough to add the minimal set of necessary and sufficient control dependencies for your code to run correctly.
---------------------------------------------------------------------------------------------------------------
'''
# Automatic control dependencies

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f3(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

print('f3(1.0, 2.0) = \n{0}\n'.format(f3(1.0, 2.0)))  # 10.0
print('f3(1.0, 2.0).shape = \n{0}\n'.format(f3(1.0, 2.0).shape))
print('f3(1.0, 2.0).dtype = \n{0}\n'.format(f3(1.0, 2.0).dtype))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Variables                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
We can use the same idea of leveraging the intended execution order of the code 
to make variable creation and utilization very easy in tf.function. 
There is one very important caveat, though, 
which is that with variables it's possible to write code which behaves differently in eager mode and graph mode.

Specifically, this will happen when you create a new Variable with each call. 
Due to tracing semantics, tf.function will reuse the same variable each call, but eager mode will create a new variable with each call. 
To guard against this mistake, tf.function will raise an error if it detects dangerous variable creation behavior.
---------------------------------------------------------------------------------------------------------------
'''
@tf.function
def f4(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

with assert_raises(ValueError):
    f4(1.0)

# Non-ambiguous code is ok though

v = tf.Variable(1.0)

@tf.function
def f5(x):
    return v.assign_add(x)

print(f5(1.0))  # 2.0
print(f5(2.0))  # 4.0

# You can also create variables inside a tf.function as long as we can prove
# that those variables are created only the first time the function is executed.

class C: pass
obj = C(); obj.v = None

@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)

print(g(1.0))  # 2.0
print(g(2.0))  # 4.0

# Variable initializers can depend on function arguments and on values of other
# variables. We can figure out the right initialization order using the same
# method we use to generate control dependencies.

state = []
@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Using AutoGraph                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The autograph library is fully integrated with tf.function, 
and it will rewrite conditionals and loops which depend on Tensors to run dynamically in the graph.

tf.cond and tf.while_loop continue to work with tf.function, 
but code with control flow is often easier to write and understand when written in imperative style.
---------------------------------------------------------------------------------------------------------------
'''
# Simple loop

@tf.function
def f6(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

f6(tf.random.uniform([5]))

# If you're curious you can inspect the code autograph generates.
# It feels like reading assembly language, though.

def f7(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f7))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       AutoGraph: Conditionals                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
AutoGraph will convert if statements into the equivalent tf.cond calls.

This substitution is made if the condition is a Tensor. 
Otherwise, the conditional is executed during tracing.
---------------------------------------------------------------------------------------------------------------
'''
def test_tf_cond(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'cond' for node in g.as_graph_def().node):
        print("{0}({1}) uses tf.cond.".format(f.__name__, ', '.join(map(str, args))))
    else:
        print("{0}({1}) executes normally.".format(f.__name__, ', '.join(map(str, args))))

@tf.function
def hyperparam_cond(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x

@tf.function
def maybe_tensor_cond(x):
    if x < 0:
        x = -x
    return x

test_tf_cond(hyperparam_cond, tf.ones([1], dtype=tf.float32))
test_tf_cond(maybe_tensor_cond, tf.constant(-1))
test_tf_cond(maybe_tensor_cond, -1)

'''
----------------------------------------------------------------------------------------
tf.cond has a number of subtleties.

* it works by tracing both sides of the conditional, 
	and then choosing the appropriate branch at runtime, depending on the condition. 
	Tracing both sides can result in unexpected execution of Python code
* it requires that if one branch creates a tensor used downstream, 
	the other branch must also create that tensor.
----------------------------------------------------------------------------------------
'''
@tf.function
def f8():
    x = tf.constant(0)
    if tf.constant(True):
        x = x + 1
        print("Tracing `then` branch")
    else:
        x = x - 1
        print("Tracing `else` branch")
    return x

f8()

@tf.function
def f9():
    if tf.constant(True):
        x = tf.ones([3, 3])
    return x

# Throws an error because both branches need to define `x`.
with assert_raises(ValueError):
    f9()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       AutoGraph and loops                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
AutoGraph has a few simple rules for converting loops.

	* for: Convert if the iterable is a tensor
	* while: Convert if the while condition depends on a tensor
	
If a loop is converted, it will be dynamically unrolled with tf.while_loop, 
or in the special case of a for x in tf.data.Dataset, transformed into tf.data.Dataset.reduce.

If a loop is not converted, it will be statically unrolled
--------------------------------------------------------------------------------------------------------------
'''
def test_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        print("{}({}) uses tf.while_loop.".format(f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        print("{}({}) uses tf.data.Dataset.reduce.".format(f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) gets unrolled.".format(f.__name__, ', '.join(map(str, args))))

@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_range)

@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_tfrange)

@tf.function
def for_in_tfdataset():
    x = tf.constant(0, dtype=tf.int64)
    for i in tf.data.Dataset.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_tfdataset)

@tf.function
def while_py_cond():
    x = 5
    while x > 0:
        x -= 1
    return x

test_dynamically_unrolled(while_py_cond)

@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x > 0:
        x -= 1
    return x


test_dynamically_unrolled(while_tf_cond)

'''
----------------------------------------------------------------------------------------
If you have a break or early return clause that depends on a tensor, 
the top-level condition or iterable should also be a tensor.
----------------------------------------------------------------------------------------
'''
# Compare the following examples:
@tf.function
def while_py_true_py_break(x):
    while True:  # py true
        if x == 0: # py break
            break
        x -= 1
    return x

test_dynamically_unrolled(while_py_true_py_break, 5)

@tf.function
def buggy_while_py_true_tf_break(x):
    while True:   # py true
        if tf.equal(x, 0): # tf break
            break
        x -= 1
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)

@tf.function
def while_tf_true_tf_break(x):
    while tf.constant(True): # tf true
        if x == 0:  # py break
            break
        x -= 1
    return x

test_dynamically_unrolled(while_tf_true_tf_break, 5)

@tf.function
def buggy_py_for_tf_break():
    x = 0
    for i in range(5):  # py for
        if tf.equal(i, 3): # tf break
            break
        x += i
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_py_for_tf_break)

@tf.function
def tf_for_py_break():
    x = 0
    for i in tf.range(5): # tf for
        if i == 3:  # py break
            break
        x += i
    return x

test_dynamically_unrolled(tf_for_py_break)

'''
---------------------------------------------------------------------------------------------------------
In order to accumulate results from a dynamically unrolled loop, you'll want to use tf.TensorArray.
---------------------------------------------------------------------------------------------------------
'''

batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
    return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    max_seq_len = input_data.shape[0]

    states = tf.TensorArray(tf.float32, size=max_seq_len)
    state = initial_state
    for i in tf.range(max_seq_len):
        state = rnn_step(input_data[i], state)
        states = states.write(i, state)
    return tf.transpose(states.stack(), [1, 0, 2])

dynamic_rnn(
                rnn_step,
                tf.random.uniform([batch_size, seq_len, feature_size]),
                tf.zeros([batch_size, feature_size])
            )

'''
-------------------------------------------------------------------------------------------------------------
As with tf.cond, tf.while_loop also comes with a number of subtleties.

	* Since a loop can execute 0 times, 
	all tensors used downstream of the while_loop must be initialized above the loop
	
	* The shape/dtypes of all loop variables must stay consistent with each iteration
------------------------------------------------------------------------------------------------------------
'''
@tf.function
def buggy_loop_var_uninitialized():
    for i in tf.range(3):
        x = i
    return x

with assert_raises(ValueError):
    buggy_loop_var_uninitialized()

@tf.function
def f10():
    x = tf.constant(0)
    for i in tf.range(3):
        x = i
    return x

print('f10() = \n{0}\n'.format(f10()))
print('f10().shape = \n{0}\n'.format(f10().shape))
print('f10().dtype = \n{0}\n'.format(f10().dtype))

@tf.function
def buggy_loop_type_changes():
    x = tf.constant(0, dtype=tf.float32)
    for i in tf.range(3): # Yields tensors of type tf.int32...
        x = i
    return x

with assert_raises(tf.errors.InvalidArgumentError):
    buggy_loop_type_changes()

@tf.function
def buggy_concat():
    x = tf.ones([0, 10])
    for i in tf.range(5):
        x = tf.concat([x, tf.ones([1, 10])], axis=0)
    return x

with assert_raises(ValueError):
    buggy_concat()

@tf.function
def concat_with_padding():
    x = tf.zeros([5, 10])
    for i in tf.range(5):
        x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4-i, 10])], axis=0)
        x.set_shape([5, 10])
    return x

print('concat_with_padding() = \n{0}\n'.format(concat_with_padding()))
print('concat_with_padding().shape = \n{0}\n'.format(concat_with_padding().shape))
print('concat_with_padding().dtype = \n{0}\n'.format(concat_with_padding().dtype))

data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print(
        '       finished        customization_Better_performance_with_tf_function.py         ({0})       \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()