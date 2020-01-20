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
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
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
                                            fname=PROJECT_ROOT_DIR.joinpath('csv_data/iris/iris_training.csv'), 
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
im = Image.open(PROJECT_ROOT_DIR.joinpath('image/Figure 3. Optimization algorithms visualized over time in 3D space.jpg'))
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

