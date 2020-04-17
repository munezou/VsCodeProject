'''
***************************************************
Matrices and Matrix Operations
---------------------------------------------------
    This function introduces various ways to create
    matrices and how to use them in TensorFlow
****************************************************
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import datetime
from packaging import version
import numpy as np
import tensorflow as tf

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Declaring matrices                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       identity_matrix                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Identity matrix
@tf.function
def  identity_matrix():
    return tf.linalg.diag(np.array([1.0,1.0,1.0]))

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir1 = os.path.join(PROJECT_ROOT_DIR, 'log/ident/{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir1)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing.
print(identity_matrix())

with writer.as_default():
    tf.summary.trace_export(
        name = "Sidentity_matrix",
        step = 0,
        profiler_outdir=logdir1
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       rundam_matrix                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def random_norm_matrix():
    return tf.random.truncated_normal(np.array([2,3]))

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir2 = os.path.join(PROJECT_ROOT_DIR, 'log/randam/{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir2)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(random_norm_matrix())

with writer.as_default():
    tf.summary.trace_export(
        name = "rundam_matrix",
        step = 0,
        profiler_outdir=logdir2
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       r2x3 constant matrix                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def fill_const_matrix():
    return tf.fill([2,3], 5.0)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir3 = os.path.join(PROJECT_ROOT_DIR, 'log/fill/{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir3)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(fill_const_matrix())

with writer.as_default():
    tf.summary.trace_export(
        name = "fill_const_matrix",
        step = 0,
        profiler_outdir=logdir3
    )

# finish to trace
tf.summary.trace_off()



'''
# 3x2 random uniform matrix
C = tf.random_uniform([3,2])
print(sess.run(C))  # Note that we are reinitializing, hence the new random variables

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/subtraction
print(sess.run(A+B))
print(sess.run(B-B))

# Matrix Multiplication
print(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose
print(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant
print(sess.run(tf.matrix_determinant(D)))

# Matrix Inverse
print(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors
print(sess.run(tf.self_adjoint_eig(D)))
'''