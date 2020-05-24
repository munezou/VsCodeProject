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
logdir1 = os.path.join(PROJECT_ROOT_DIR, 'log', 'ident', '{}'.format(stamp))
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
logdir2 = os.path.join(PROJECT_ROOT_DIR, 'log', 'randam', '{}'.format(stamp))
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
logdir3 = os.path.join(PROJECT_ROOT_DIR, 'log', 'fill', '{}'.format(stamp))
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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create matrix from np array                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

@tf.function
def create_matrix_np_array():
    x = np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]])
    return tf.convert_to_tensor(x)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir4 = os.path.join(PROJECT_ROOT_DIR, 'log', 'create_tensor', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir4)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(create_matrix_np_array())

with writer.as_default():
    tf.summary.trace_export(
        name = "create_tensor_matrix",
        step = 0,
        profiler_outdir=logdir4
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix addition                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def matrix_addition():
    A = tf.random.truncated_normal(np.array([2, 3]))
    B = tf.fill([2, 3], 5.0)
    return A + B

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir5 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_addition', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir5)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(matrix_addition())

with writer.as_default():
    tf.summary.trace_export(
        name = "matrix_addition",
        step = 0,
        profiler_outdir=logdir5
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix subtraction                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def matrix_subtraction():
    A = tf.random.truncated_normal(np.array([2, 3]))
    B = tf.fill([2, 3], 5.0)
    return A - B

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir6 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_subtraction', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir6)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(matrix_subtraction())

with writer.as_default():
    tf.summary.trace_export(
        name = "matrix_substraction",
        step = 0,
        profiler_outdir=logdir6
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix Multiplication                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def multiplication_matrix():
    A = tf.fill([2, 3], 5.0)
    B = tf.linalg.diag(np.array([1.0, 1.0, 1.0]))
    identity_matrix = tf.cast(B, tf.float32)
    return tf.linalg.matmul(A, identity_matrix)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir7 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_multiplication', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir7)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(multiplication_matrix())

with writer.as_default():
    tf.summary.trace_export(
        name = "multiplication_matrix",
        step = 0,
        profiler_outdir=logdir7
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix Transpose                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def matrix_transpose():
    C = tf.random.uniform([3, 2])
    return tf.transpose(C)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir8 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_transpose', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir8)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(matrix_transpose())

with writer.as_default():
    tf.summary.trace_export(
        name = "multiplication_matrix",
        step = 0,
        profiler_outdir=logdir8
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix Determinant                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def matrix_determinant():
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
    return tf.linalg.det(D)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir9 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_determinant', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir9)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(matrix_determinant())

with writer.as_default():
    tf.summary.trace_export(
        name = "matrix_determinant",
        step = 0,
        profiler_outdir=logdir9
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Matrix Inverse                                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def matrix_inverse():
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
    return tf.linalg.inv(D)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir10 = os.path.join(PROJECT_ROOT_DIR, 'log', 'matrix_inverse', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir10)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(matrix_determinant())

with writer.as_default():
    tf.summary.trace_export(
        name = "matrix_inverse",
        step = 0,
        profiler_outdir=logdir10
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Cholesky Decomposition                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def cholesky_decomposition():
    B = tf.linalg.diag(np.array([1.0, 1.0, 1.0]))
    identity_matrix = tf.cast(B, tf.float32)
    return tf.linalg.cholesky(identity_matrix)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir11 = os.path.join(PROJECT_ROOT_DIR, 'log', 'cholesky_decomposition', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir11)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(cholesky_decomposition())

with writer.as_default():
    tf.summary.trace_export(
        name = "cholesky_decomposition",
        step = 0,
        profiler_outdir=logdir11
    )

# finish to trace
tf.summary.trace_off()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Eigenvalues and Eigenvectors                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
@tf.function
def Eigenvalues_Eigenvectors():
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
    return tf.linalg.eig(D)

# set up logging
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir12 = os.path.join(PROJECT_ROOT_DIR, 'log', 'Eigenvalues_Eigenvectors', '{}'.format(stamp))
writer = tf.summary.create_file_writer(logdir12)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)

print(Eigenvalues_Eigenvectors())

with writer.as_default():
    tf.summary.trace_export(
        name = "Eigenvalues_Eigenvectors",
        step = 0,
        profiler_outdir=logdir12
    )

# finish to trace
tf.summary.trace_off()

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         matrices.py                                  ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()