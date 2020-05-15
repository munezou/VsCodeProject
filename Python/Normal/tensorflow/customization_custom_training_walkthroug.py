'''
------------------------------------------------------------------------------------------
customization
    Custom training: walkthrough

This guide uses machine learning to categorize Iris flowers by species. It uses TensorFlow to:

1. Build a model,
2. Train this model on example data, and
3. Use the model to make predictions about unknown data.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
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
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       TensorFlow programming                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
This guide uses these high-level TensorFlow concepts:

* Use TensorFlow's default eager execution development environment,
* Import data with the Datasets API,
* Build models and layers with TensorFlow's Keras API.

This tutorial is structured like many TensorFlow programs:
1. Import and parse the dataset.
2. Select the type of model.
3. Train the model.
4. Evaluate the model's effectiveness.
5. Use the trained model to make predictions.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Configure imports                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Import TensorFlow and the other required Python modules. 
By default, TensorFlow uses eager execution to evaluate operations immediately, 
returning concrete values instead of creating a computational graph that is executed later. 
If you are used to a REPL or the python interactive console, this feels familiar.
---------------------------------------------------------------------------------------------------------------
'''
print("Eager execution: {}".format(tf.executing_eagerly()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The Iris classification problem                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Imagine you are a botanist seeking an automated way to categorize each Iris flower you find. 
Machine learning provides many algorithms to classify flowers statistically. 
For instance, a sophisticated machine learning program could classify flowers based on photographs. 
Our ambitions are more modest—we're going to classify Iris flowers based on the length 
and width measurements of their sepals and petals.

The Iris genus entails about 300 species, but our program will only classify the following three:

* Iris setosa
* Iris virginica
* Iris versicolor
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images', 'iris_three_species.jpg'))
im.show()

'''
-----------------------------------------------------------------------------------------------------------------
Fortunately, someone has already created a dataset of 120 Iris flowers with the sepal and petal measurements. 
This is a classic dataset that is popular for beginner machine learning classification problems.
-----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Import and parse the training dataset                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Download the dataset file and convert it into a structure that can be used by this Python program.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the dataset                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Download the training dataset file using the tf.keras.utils.get_file function. 
This returns the file path of the downloaded file:
---------------------------------------------------------------------------------------------------------------
'''
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(
                                            origin=train_dataset_url,
                                            fname=PROJECT_ROOT_DIR.joinpath('csv_data', 'iris', 'iris_training.csv'), 
                                        )

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {0}".format(feature_names))
print("Label: {0}\n".format(label_name))

'''
-------------------------------------------------------------------------------------------------------------
Each label is associated with string name (for example, "setosa"), 
but machine learning typically relies on numeric values. 
The label numbers are mapped to a named representation, such as:

0: Iris setosa
1: Iris versicolor
2: Iris virginica
-------------------------------------------------------------------------------------------------------------
'''
# For more information about features and labels, see the ML Terminology section of the Machine Learning Crash Course.
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create a tf.data.Dataset                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
TensorFlow's Dataset API handles many common cases for loading data into a model. 
This is a high-level API for reading data and transforming it into a form used for training. 
See the Datasets Quick Start guide for more information.

Since the dataset is a CSV-formatted text file, 
use the make_csv_dataset function to parse the data into a suitable format. 
Since this function generates data for training models, the default behavior is to shuffle the data (shuffle=True, shuffle_buffer_size=10000), 
and repeat the dataset forever (num_epochs=None). We also set the batch_size parameter:
---------------------------------------------------------------------------------------------------------------
'''
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
                        train_dataset_fp,
                        batch_size,
                        column_names=column_names,
                        label_name=label_name,
                        num_epochs=1
                    )

'''
----------------------------------------------------------------------------------------------------------------
The make_csv_dataset function returns a tf.data.Dataset of (features, label) pairs, 
where features is a dictionary: {'feature_name': value}

These Dataset objects are iterable. Let's look at a batch of features:
----------------------------------------------------------------------------------------------------------------
'''

features, labels = next(iter(train_dataset))

print(features)

'''
----------------------------------------------------------------------------------------------------------------
Notice that like-features are grouped together, 
or batched. Each example row's fields are appended to the corresponding feature array. 
Change the batch_size to set the number of examples stored in these feature arrays.
----------------------------------------------------------------------------------------------------------------
'''
# You can start to see some clusters by plotting a few features from the batch:
plt.scatter(
                features['petal_length'],
                features['sepal_length'],
                c=labels,
                cmap='viridis'
            )

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

'''
-----------------------------------------------------------------------------------------------------------------
To simplify the model building step, 
create a function to repackage the features dictionary into a single array with shape: (batch_size, num_features).

This function uses the tf.stack method which takes values from a list of tensors and creates a combined tensor 
at the specified dimension:
-----------------------------------------------------------------------------------------------------------------
'''
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

'''
----------------------------------------------------------------------------------------------------------------
Then use the tf.data.Dataset.map method to pack the features of each (features,label) pair into the training dataset:
----------------------------------------------------------------------------------------------------------------
'''
train_dataset = train_dataset.map(pack_features_vector)

'''
----------------------------------------------------------------------------------------------------------------
The features element of the Dataset are now arrays with shape (batch_size, num_features). 
Let's look at the first few examples:
----------------------------------------------------------------------------------------------------------------
'''
features, labels = next(iter(train_dataset))

print(features[:5])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Select the type of model                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Why model?
A model is a relationship between features and the label. 
For the Iris classification problem, the model defines the relationship between the sepal and petal measurements and the predicted Iris species. 
Some simple models can be described with a few lines of algebra, 
but complex machine learning models have a large number of parameters that are difficult to summarize.

Could you determine the relationship between the four features and the Iris species without using machine learning? That is, 
could you use traditional programming techniques (for example, a lot of conditional statements) to create a model? 
Perhaps—if you analyzed the dataset long enough to determine the relationships between petal and sepal measurements to a particular species. 
And this becomes difficult—maybe impossible—on more complicated datasets. 
A good machine learning approach determines the model for you. 
If you feed enough representative examples into the right machine learning model type, the program will figure out the relationships for you.


Select the model
We need to select the kind of model to train. 
There are many types of models and picking a good one takes experience. 
This tutorial uses a neural network to solve the Iris classification problem. 
Neural networks can find complex relationships between features and the label. 
It is a highly-structured graph, organized into one or more hidden layers. 
Each hidden layer consists of one or more neurons. 
There are several categories of neural networks and this program uses a dense, 
or fully-connected neural network: the neurons in one layer receive input connections from every neuron in the previous layer. 
For example, Figure 2 illustrates a dense neural network consisting of an input layer, two hidden layers, and an output layer:
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images', 'Figure 2.png'))
im.show()

'''
---------------------------------------------------------------------------------------------------------------
When the model from Figure 2 is trained and fed an unlabeled example, 
it yields three predictions: the likelihood that this flower is the given Iris species. 
This prediction is called inference. 
For this example, the sum of the output predictions is 1.0. In Figure 2, 
this prediction breaks down as: 0.02 for Iris setosa, 0.95 for Iris versicolor, and 0.03 for Iris virginica. 
This means that the model predicts—with 95% probability—that an unlabeled example flower is an Iris versicolor.
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create a model using Keras                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The TensorFlow tf.keras API is the preferred way to create models and layers. 
This makes it easy to build models and experiment while Keras handles the complexity of connecting everything together.

The tf.keras.Sequential model is a linear stack of layers. 
Its constructor takes a list of layer instances, in this case, two Dense layers with 10 nodes each, 
and an output layer with 3 nodes representing our label predictions. 
The first layer's input_shape parameter corresponds to the number of features from the dataset, and is required:
--------------------------------------------------------------------------------------------------------------
'''

model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(3)
        ])

'''
-------------------------------------------------------------------------------------------------------------
The activation function determines the output shape of each node in the layer. 
These non-linearities are important—without them the model would be equivalent to a single layer. 
There are many available activations, but ReLU is common for hidden layers.

The ideal number of hidden layers and neurons depends on the problem and the dataset. 
Like many aspects of machine learning, 
picking the best shape of the neural network requires a mixture of knowledge and experimentation. 
As a rule of thumb, increasing the number of hidden layers and neurons typically creates a more powerful model, 
which requires more data to train effectively.
-------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Using the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Let's have a quick look at what this model does to a batch of features:
predictions = model(features)
predictions[:5]

'''
--------------------------------------------------------------------------------------------------------------
Here, each example returns a logit for each class.

To convert these logits to a probability for each class, use the softmax function:
--------------------------------------------------------------------------------------------------------------
'''

tf.nn.softmax(predictions[:5])

'''
--------------------------------------------------------------------------------------------------------------
Taking the tf.argmax across classes gives us the predicted class index. 
But, the model hasn't been trained yet, so these aren't good predictions:
--------------------------------------------------------------------------------------------------------------
'''
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
Training is the stage of machine learning when the model is gradually optimized, or the model learns the dataset. 
The goal is to learn enough about the structure of the training dataset to make predictions about unseen data. 
If you learn too much about the training dataset, 
then the predictions only work for the data it has seen and will not be generalizable. 
This problem is called *overfitting*—it's like memorizing the answers instead of understanding how to solve a problem.

The Iris classification problem is an example of supervised machine learning: the model is trained from examples that contain labels. 
In unsupervised machine learning, the examples don't contain labels. Instead, the model typically finds patterns among the features.
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Define the loss and gradient function                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Both training and evaluation stages need to calculate the model's loss. 
This measures how off a model's predictions are from the desired label, in other words, how bad the model is performing. 
We want to minimize, or optimize, this value.

Our model will calculate its loss using the tf.keras.losses.SparseCategoricalCrossentropy function 
which takes the model's class probability predictions and the desired label, 
and returns the average loss across the examples.
--------------------------------------------------------------------------------------------------------------
'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))

# Use the tf.GradientTape context to calculate the gradients used to optimize your model:
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create an optimizer                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
An optimizer applies the computed gradients to the model's variables to minimize the loss function. 
You can think of the loss function as a curved surface (see Figure 3) and we want to find its lowest point by walking around. 
The gradients point in the direction of steepest ascent—so we'll travel the opposite way and move down the hill. 
By iteratively calculating the loss and gradient for each batch, we'll adjust the model during training. 
Gradually, the model will find the best combination of weights and bias to minimize loss. 
And the lower the loss, the better the model's predictions.
---------------------------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images', 'Figure 3. Optimization algorithms visualized over time in 3D space.jpg'))
im.show()

'''
----------------------------------------------------------------------------------------------------------------
TensorFlow has many optimization algorithms available for training. 
This model uses the tf.keras.optimizers.SGD that implements the stochastic gradient descent (SGD) algorithm. 
The learning_rate sets the step size to take for each iteration down the hill. 
This is a hyperparameter that you'll commonly adjust to achieve better results.
----------------------------------------------------------------------------------------------------------------
'''
# Let's setup the optimizer:
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

'''
----------------------------------------------------------------------------------------------------------------
We'll use this to calculate a single optimization step:
----------------------------------------------------------------------------------------------------------------
'''
loss_value, grads = grad(model, features, labels)

print("Step: {0}, Initial Loss: {1}".format(optimizer.iterations.numpy(), loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {0},         Loss: {1}".format(optimizer.iterations.numpy(), loss(model, features, labels).numpy()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Training loop                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
With all the pieces in place, the model is ready for training! 
A training loop feeds the dataset examples into the model to help it make better predictions. 
The following code block sets up these training steps:

1. Iterate each epoch. An epoch is one pass through the dataset.
2. Within an epoch, iterate over each example in the training Dataset grabbing its features (x) and label (y).
3. Using the example's features, make a prediction and compare it with the label. 
   Measure the inaccuracy of the prediction and use that to calculate the model's loss and gradients.
4. Use an optimizer to update the model's variables.
5. Keep track of some stats for visualization.
6. Repeat for each epoch.

The num_epochs variable is the number of times to loop over the dataset collection. 
Counter-intuitively, training a model longer does not guarantee a better model. 
num_epochs is a hyperparameter that you can tune. 
Choosing the right number usually requires both experience and experimentation:
----------------------------------------------------------------------------------------------------------------
'''

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        epoch_accuracy(y, model(x))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Visualize the loss function over time                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
While it's helpful to print out the model's training progress, it's often more helpful to see this progress. 
TensorBoard is a nice visualization tool that is packaged with TensorFlow, 
but we can create basic charts using the matplotlib module.

Interpreting these charts takes some experience, 
but you really want to see the loss go down and the accuracy go up:
---------------------------------------------------------------------------------------------------------------
'''
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished         customization_custom_training_walkthroug.py            (2020/05/16)           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()
