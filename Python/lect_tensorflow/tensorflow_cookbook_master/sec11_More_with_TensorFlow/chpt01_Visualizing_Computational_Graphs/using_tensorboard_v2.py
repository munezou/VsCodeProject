# -*- coding: utf-8 -*-
# Using Tensorboard
#----------------------------------
#
# We illustrate the various ways to use
#  Tensorboard

import os
import io
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Initialize a graph session
sess = tf.compat.v1.Session()

# Create tensorboard folder if not exists
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
print('Running a slowed down linear regression. '
      'Run the command: $tensorboard --logdir="tensorboard"  '
      ' Then navigate to http://127.0.0.1:6006')

# You can also specify a port option with --port 6006

# Create a visualizer object
summary_writer = tf.compat.v1.summary.FileWriter('tensorboard', sess.graph)

# Wait a few seconds for user to run tensorboard commands
time.sleep(3)

# Some parameters
batch_size = 50
generations = 100

# Create sample input data
x_data = np.arange(1000)/10.
true_slope = 2.
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

# Split into train/test
train_ix = np.random.choice(len(x_data), size=int(len(x_data)*0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], y_data[test_ix]

# Declare placeholders
x_graph_input = tf.compat.v1.placeholder(tf.float32, [None])
y_graph_input = tf.compat.v1.placeholder(tf.float32, [None])

# Declare model variables
m = tf.Variable(tf.random.normal([1], dtype=tf.float32), name='Slope')

# Declare model
output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')

# Declare loss function (L1)
residuals = output - y_graph_input
l1_loss = tf.reduce_mean(input_tensor=tf.abs(residuals), name="L1_Loss")

# Declare optimization function
my_optim = tf.compat.v1.train.GradientDescentOptimizer(0.01)
train_step = my_optim.minimize(l1_loss)

# Visualize a scalar
with tf.compat.v1.name_scope('Slope_Estimate'):
    tf.compat.v1.summary.scalar('Slope_Estimate', tf.squeeze(m))
    
# Visualize a histogram (errors)
with tf.compat.v1.name_scope('Loss_and_Residuals'):
    tf.compat.v1.summary.histogram('Histogram_Errors', tf.squeeze(l1_loss))
    tf.compat.v1.summary.histogram('Histogram_Residuals', tf.squeeze(residuals))


# Declare summary merging operation
summary_op = tf.compat.v1.summary.merge_all()

# Initialize Variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for i in range(generations):
    batch_indices = np.random.choice(len(x_data_train), size=batch_size)
    x_batch = x_data_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, train_loss, summary = sess.run([train_step, l1_loss, summary_op],
                             feed_dict={x_graph_input: x_batch,
                                        y_graph_input: y_batch})
    
    test_loss, test_resids = sess.run([l1_loss, residuals], feed_dict={x_graph_input: x_data_test,
                                                                       y_graph_input: y_data_test})
    
    if (i + 1) % 10 == 0:
        print('Generation {} of {}. Train Loss: {:.3}, Test Loss: {:.3}.'.format(i+1, generations, train_loss, test_loss))

    log_writer = tf.compat.v1.summary.FileWriter('tensorboard')
    log_writer.add_summary(summary, i)
    time.sleep(0.5)

#Create a function to save a protobuf bytes version of the graph
def gen_linear_plot(slope):
    linear_prediction = x_data * slope
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return(buf)

# Add image to tensorboard (plot the linear fit!)
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
# Add image summary
image_summary_op = tf.compat.v1.summary.image("Linear_Plot", image)
image_summary = sess.run(image_summary_op)
log_writer.add_summary(image_summary, i)
log_writer.close()
