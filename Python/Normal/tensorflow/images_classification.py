'''
------------------------------------------------------------------------------------------
images
    Image classification

This tutorial shows how to classify cats or dogs from images. 
It builds an image classifier using a tf.keras.Sequential model 
and load data using tf.keras.preprocessing.image.ImageDataGenerator. 
You will get some practical experience and develop intuition for the following concepts:

	* Building data input pipelines using the tf.keras.preprocessing.image.ImageDataGenerator class 
	to efficiently work with data on disk to use with the model.
	
	* Overfitting —How to identify and prevent it.
	
	* Data augmentation and dropout —Key techniques to fight overfitting in computer vision tasks 
	to incorporate into the data pipeline and image classifier model.
	
This tutorial follows a basic machine learning workflow:

	1. Examine and understand data
	2. Build an input pipeline
	3. Build the model
	4. Train the model
	5. Test the model
	6. Improve the model and repeat the process

Import packages

Let's start by importing the required packages. 
The os package is used to read files and directory structure, 
NumPy is used to convert python list to numpy array and to perform required matrix operations and matplotlib.pyplot 
to plot the graph and display images in the training and validation data.
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        '       Download and prepare the CIFAR10 dataset                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

fName = PROJECT_ROOT_DIR.joinpath('Data', 'cats_and_dog.zip')

cacheDir = PROJECT_ROOT_DIR.joinpath('Data')

path_to_zip = tf.keras.utils.get_file(
                                        origin=_URL,
                                        fname=fName, 
                                        extract=True,
                                        cache_dir=cacheDir
                                    )

PATH = os.path.join(os.path.dirname(path_to_zip), 'datasets', 'cats_and_dogs_filtered')

'''
--------------------------------------------------------------------------------------------------------------
The dataset has the following directory structure:

cats_and_dogs_filtered
|__ train
    |______ cats: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
    |______ dogs: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
|__ validation
    |______ cats: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
After extracting its contents, assign variables with the proper file path for the training and validation set.
-----------------------------------------------------------------------------------------------------------------
'''
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Understand the data                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Let's look at how many cats and dogs images are in the training and validation directory:
--------------------------------------------------------------------------------------------------------------
'''
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

'''
------------------------------------------------------------------------------------------------------------------
For convenience, set up variables to use while pre-processing the dataset and training the network.
------------------------------------------------------------------------------------------------------------------
'''
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Data preparation                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Format the images into appropriately pre-processed floating point tensors 
before feeding to the network:

	* Read images from the disk.
	
	* Decode contents of these images and convert it into proper grid format 
	as per their RGB content.
	
	* Convert them into floating point tensors.
	
	* Rescale the tensors from values between 0 and 255 to values between 0 and 1, 
	as neural networks prefer to deal with small input values.
	
Fortunately, 
all these tasks can be done with the ImageDataGenerator class provided by tf.keras. 
It can read images from disk and preprocess them into proper tensors. 
It will also set up generators that convert these images into batches of tensors—helpful 
when training the network.
------------------------------------------------------------------------------------------------------------------
'''
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

'''
------------------------------------------------------------------------------------------------------------------
After defining the generators for training and validation images, 
the flow_from_directory method load images from the disk, 
applies rescaling, and resizes the images into the required dimensions.
------------------------------------------------------------------------------------------------------------------
'''
train_data_gen = train_image_generator.flow_from_directory(
                    batch_size=batch_size,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='binary'
                )

val_data_gen = validation_image_generator.flow_from_directory(
                    batch_size=batch_size,
                    directory=validation_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='binary'
                )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Visualize training images                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Visualize the training images by extracting a batch of images 
from the training generator—which is 32 images in this example—then 
plot five of them with matplotlib.
--------------------------------------------------------------------------------------------------------------
'''
sample_training_images, _ = next(train_data_gen)

'''
-------------------------------------------------------------------------------------------------------------
The next function returns a batch from the dataset. 
The return value of next function is in form of (x_train, y_train) 
where x_train is training features and y_train, its labels. 
Discard the labels to only visualize the training images.
------------------------------------------------------------------------------------------------------------
'''
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the model                                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
The model consists of three convolution blocks with a max pool layer in each of them. 
There's a fully connected layer with 512 units on top of it that is activated by a relu activation function. 
The model outputs class probabilities based on binary classification by the sigmoid activation function.
--------------------------------------------------------------------------------------------------------------
'''
model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Compile the model                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
For this tutorial, 
choose the ADAM optimizer and binary cross entropy loss function. 
To view training and validation accuracy for each training epoch, 
pass the metrics argument.
---------------------------------------------------------------------------------------------------------------
'''
model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Model summary                                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
View all the layers of the network using the model's summary method:
---------------------------------------------------------------------------------------------------------------
'''
model.summary()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Use the fit_generator method of the ImageDataGenerator class to train the network.
--------------------------------------------------------------------------------------------------------------
'''
history = model.fit_generator(
                train_data_gen,
                steps_per_epoch=total_train // batch_size,
                epochs=epochs,
                validation_data=val_data_gen,
                validation_steps=total_val // batch_size
            )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Visualize training results                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Now visualize the results after training the network.
---------------------------------------------------------------------------------------------------------------
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

'''
----------------------------------------------------------------------------------------------------------------
As you can see from the plots, 
training accuracy and validation accuracy are off by large margin 
and the model has achieved only around 70% accuracy on the validation set.

Let's look at what went wrong and try to increase overall performance of the model.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Overfitting                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
In the plots above, 
the training accuracy is increasing linearly over time, 
whereas validation accuracy stalls around 70% in the training process. 
Also, 
the difference in accuracy between training and validation accuracy is noticeable—a sign of overfitting.

When there are a small number of training examples, 
the model sometimes learns from noises or unwanted details from training examples—to an extent that it negatively impacts the performance of the model on new examples. 
This phenomenon is known as overfitting. 
It means that the model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In this tutorial, you'll use data augmentation and add dropout to our model.
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Data augmentation                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Overfitting generally occurs when there are a small number of training examples. 
One way to fix this problem is to augment the dataset so that it has a sufficient number of training examples. 
Data augmentation takes the approach of generating more training data from existing training samples by augmenting the samples using random transformations that yield believable-looking images. 
The goal is the model will never see the exact same picture twice during training. 
This helps expose the model to more aspects of the data and generalize better.

Implement this in tf.keras using the ImageDataGenerator class. 
Pass different transformations to the dataset and it will take care of applying it during the training process.
--------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Augment and visualize data                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Begin by applying random horizontal flip augmentation to the dataset 
and see how individual images look like after the transformation.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Apply horizontal flip                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Pass horizontal_flip as an argument to the ImageDataGenerator class and set it to True to apply this augmentation.
---------------------------------------------------------------------------------------------------------------
'''
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(
                    batch_size=batch_size,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH)
                )

'''
-------------------------------------------------------------------------------------------------------------
Take one sample image from the training examples and repeat it five times 
so that the augmentation is applied to the same image five times.
-------------------------------------------------------------------------------------------------------------
'''
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Re-use the same custom plotting function defined and used
# above to visualize the training images
plotImages(augmented_images)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Randomly rotate the image                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Let's take a look at a different augmentation called rotation and apply 45 degrees of rotation randomly to the training examples.
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)

train_data_gen = image_gen.flow_from_directory(
                    batch_size=batch_size,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH)
                )

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Apply zoom augmentation                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Apply a zoom augmentation to the dataset to zoom images up to 50% randomly.
# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(
                    batch_size=batch_size,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH)
                )

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Put it all together                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Apply all the previous augmentations. 
# Here, you applied rescale, 45 degree rotation, width shift, height shift, horizontal flip 
# and zoom augmentation to the training images.
image_gen_train = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=45,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        zoom_range=0.5
                    )

train_data_gen = image_gen_train.flow_from_directory(
                    batch_size=batch_size,
                    directory=train_dir,
                    shuffle=True,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='binary'
                )

'''
-------------------------------------------------------------------------------------------------------
Visualize how a single image would look five different times 
when passing these augmentations randomly to the dataset.
-------------------------------------------------------------------------------------------------------
'''
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create validation data generator                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Generally, only apply data augmentation to the training examples. 
In this case, only rescale the validation images and convert them into batches using ImageDataGenerator.
--------------------------------------------------------------------------------------------------------------
'''
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
                    batch_size=batch_size,
                    directory=validation_dir,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode='binary'
                )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dropout                                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Another technique to reduce overfitting is to introduce dropout to the network. 
It is a form of regularization that forces the weights in the network to take only small values, 
which makes the distribution of weight values more regular and the network can reduce overfitting on small training examples. 
Dropout is one of the regularization technique used in this tutorial

When you apply dropout to a layer it randomly drops out (set to zero) number of output units from the applied layer during the training process. 
Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. 
This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.

When appling 0.1 dropout to a certain layer, it randomly kills 10% of the output units in each training epoch.

Create a network architecture with this new dropout feature and apply it to different convolutions and fully-connected layers.
----------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Creating a new network with Dropouts                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Here, you apply dropout to first and last max pool layers. 
Applying dropout will randomly set 20% of the neurons to zero during each training epoch. 
This helps to avoid overfitting on the training dataset.
---------------------------------------------------------------------------------------------------------------
'''
model_new = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Compile the model                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# After introducing dropouts to the network, compile the model and view the layers summary.
model_new.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_new.summary()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# After successfully introducing data augmentations to the training examples and adding dropouts to the network, 
# train this new network:
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Visualize the model                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Visualize the new model after training, you can see that there is significantly less overfitting than before. 
# The accuracy should go up after training the model for more epochs.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        images_classification.py       　　       　                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()