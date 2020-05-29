'''
# Illustration of Various Kernels
#----------------------------------
#
# This function wll illustrate how to
# implement various kernels in TensorFlow.
#
# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# Generate non-lnear data
(x_vals, y_vals) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
y_vals = np.array([1 if y==1 else -1 for y in y_vals])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

plt.scatter(class1_x, class1_y, c='red', marker='x')
plt.scatter(class2_x, class2_y, c='blue', marker='o')
plt.title('raw data for analizing')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Declare batch size
batch_size = 350

# Create variables for svm
b = tf.cast(tf.Variable(tf.random.normal(shape=[1, batch_size])), dtype=tf.float32)
Gam = -50.0
# Apply kernel
# Linear Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

def kernel_loss(input_x, bb, g, targets):
    # Gaussian (RBF) kernel
    gamma = tf.cast(tf.constant(g), dtype=tf.float32)
    dist = tf.reduce_sum(input_tensor=tf.square(input_x), axis=1)
    dist = tf.reshape(dist, [-1,1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(input_x, tf.transpose(a=input_x)))), tf.transpose(a=dist))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    first_term = tf.reduce_sum(input_tensor=bb)
    b_vec_cross = tf.matmul(tf.transpose(a=bb), bb)
    target_cross = tf.matmul(targets, tf.transpose(a=targets))
    second_term = tf.reduce_sum(input_tensor=tf.multiply(my_kernel, tf.multiply(b_vec_cross, target_cross)))
    return tf.negative(tf.subtract(first_term, second_term))

def kernel_grad(input_x, bb, g, targets):
    with tf.GradientTape() as tape:
        loss_value = kernel_loss(input_x, bb, g, targets)
    return tape.gradient(loss_value, [bb])

# Create Prediction Kernel
# Linear prediction kernel
# my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

def predection(input_x, bb, g, targets, pred_grid):
    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(input_x), 1),[-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(pred_grid), 1),[-1,1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(input_x, tf.transpose(pred_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(g, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(tf.transpose(targets), bb), pred_kernel)
    return tf.sign(prediction_output-tf.reduce_mean(prediction_output))


def accuracy(input_x, bb, g, targets, pred_grid):
    prediction_value =  predection(input_x, bb, g, targets, pred_grid)
    return tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction_value), tf.squeeze(targets)), tf.float32))

# Declare optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.002)

# Training loop
loss_vec = []
batch_accuracy = []

for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = tf.cast(x_vals[rand_index], dtype=tf.float32)
    rand_y = tf.cast(np.transpose([y_vals[rand_index]]), dtype=tf.float32)
    
    grads = kernel_grad(rand_x, b, Gam, rand_y)
    optimizer.apply_gradients(zip(grads, [b]))
    
    temp_loss = kernel_loss(rand_x, b, Gam, rand_y)
    loss_vec.append(temp_loss)
    
    acc_temp = accuracy(rand_x, b, Gam, rand_y, rand_x)
    batch_accuracy.append(acc_temp)
    
    if (i +1 ) % 250 == 0:
        print('Step #{0}'.format(i + 1))
        print('Loss = {0}\n'.format(temp_loss.numpy()))

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1

xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )

grid_points = tf.cast(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32)

[grid_predictions] = predection(rand_x, b, Gam, rand_y, grid_points).numpy()

grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# Evaluate on new/unseen data points
# New data points:
new_points = np.array([
                (-0.75, -0.75),
                (-0.5, -0.5),
                (-0.25, -0.25),
                (0.25, 0.25),
                (0.5, 0.5),
                (0.75, 0.75)
            ])

[evaluations] = predection(
                    tf.cast(x_vals, dtype=tf.float32), 
                    b, 
                    Gam, 
                    tf.cast(np.transpose([y_vals]), dtype=tf.float32), 
                    tf.cast(new_points, dtype=tf.float32)
                ).numpy()


for ix, p in enumerate(new_points):
    print('p = {0} : class={1}'.format(p, evaluations[ix]))

date_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished         svm_kernels.py                                          ({0})             \n'.format(date_today)
    )

print(
        '------------------------------------------------------------------------------------------------------\n'
    )
print()
print()
print()