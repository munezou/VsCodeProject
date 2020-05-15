'''
------------------------------------------------------------------------------------------
images
    Transfer learning with a pretrained ConvNet

In this tutorial you will learn how to classify cats vs dogs images by using transfer learning from a pre-trained network.

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. 
You either use the pretrained model as it is, or use transfer learning to customize this model to a given task.

The intuition behind transfer learning is that if a model trained on a large and general enough dataset, 
this model will effectively serve as a generic model of the visual world. 
You can then take advantage of these learned feature maps without having to start from scratch training a large model on a large dataset.

In this notebook, you will try two ways to customize a pretrained model:

	1. Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. 
	You simply add a new classifier, which will be trained from scratch, 
	on top of the pretrained model so that you can repurpose the feature maps learned previously for our dataset.

	You do not need to (re)train the entire model. 
	The base convolutional network already contains features that are generically useful for classifying pictures. 
	However, the final, classification part of the pretrained model is specific to original classification task, 
	and subsequently specific to the set of classes on which the model was trained.

	2. Fine-Tuning: Unfreezing a few of the top layers of a frozen model base and jointly training both the newly-added classifier layers 
	and the last layers of the base model. 
	This allows us to "fine tune" the higher-order feature representations in the base model 
	in order to make them more relevant for the specific task.

You will follow the general machine learning workflow.

	1. Examine and understand the data
	2. Build an input pipeline, in this case using Keras ImageDataGenerator
	3. Compose our model
		* Load in our pretrained base model (and pretrained weights)
		* Stack our classification layers on top
	4. Train our model
	5. Evaluate model
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
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers
print(__doc__)

keras = tf.keras
tfds.disable_progress_bar()

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
        '       Data preprocessing                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Data download                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use TensorFlow Datasets to load the cats and dogs dataset.

This tfds package is the easiest way to load pre-defined data. 
If you have your own data, and are interested in importing using it with TensorFlow see loading image data

The tfds.load method downloads and caches the data, and returns a tf.data.Dataset object. 
These objects provide powerful, efficient methods for manipulating data and piping it into your model.

Since "cats_vs_dog" doesn't define standard splits, 
use the subsplit feature to divide it into (train, validation, test) with 80%, 10%, 10% of the data respectively.
---------------------------------------------------------------------------------------------------------------
'''
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
														'cats_vs_dogs', 
														data_dir=PROJECT_ROOT_DIR.joinpath('Data'),
														split=list(splits),
														with_info=True, 
														as_supervised=True
													)

'''
---------------------------------------------------------------------------------------------------------------
The resulting tf.data.Dataset objects contain (image, label) pairs. 
Where the images have variable shape and 3 channels, and the label is a scalar.
---------------------------------------------------------------------------------------------------------------
'''
print(raw_train)
print(raw_validation)
print(raw_test)

# Show the first two images and labels from the training set:
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
	plt.figure()
	plt.imshow(image)
	plt.title(get_label_name(label))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Format the Data                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use the tf.image module to format the images for the task.

Resize the images to a fixes input size, and rescale the input channels to a range of [-1,1]
---------------------------------------------------------------------------------------------------------------
'''
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
	image = tf.cast(image, tf.float32)
	image = (image/127.5) - 1
	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
	return image, label

# Apply this function to each item in the dataset using the map method:
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Now shuffle and batch the data.
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Inspect a batch of data:
for image_batch, label_batch in train_batches.take(1):
	pass

image_batch.shape

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create the base model from the pre-trained convnets                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
You will create the base model from the MobileNet V2 model developed at Google. 
This is pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images. 
ImageNet has a fairly arbitrary research training dataset with categories like jackfruit and syringe, 
but this base of knowledge will help us tell apart cats and dogs from our specific dataset.

First, you need to pick which layer of MobileNet V2 you will use for feature extraction. 
Obviously, the very last classification layer 
(on "top", as most diagrams of machine learning models go from bottom to top) is not very useful. 
Instead, you will follow the common practice to instead depend on the very last layer before the flatten operation. 
This layer is called the "bottleneck layer". 
The bottleneck features retain much generality as compared to the final/top layer.

First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. 
By specifying the include_top=False argument, 
you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.
-----------------------------------------------------------------------------------------------------------------
'''
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
				input_shape=IMG_SHAPE,
				include_top=False,
				weights='imagenet'
			)

'''
------------------------------------------------------------------------------------------------------------------
This feature extractor converts each 160x160x3 image to a 5x5x1280 block of features. 
See what it does to the example batch of images:
------------------------------------------------------------------------------------------------------------------
'''
feature_batch = base_model(image_batch)
print('feature_batch.shape = {0}\n'.format(feature_batch.shape))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Feature extraction                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
You will freeze the convolutional base created from the previous step and use that as a feature extractor, 
add a classifier on top of it and train the top-level classifier.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Freeze the convolutional base                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
It's important to freeze the convolutional based before you compile and train the model. 
By freezing (or setting layer.trainable = False), 
you prevent the weights in a given layer from being updated during training. 
MobileNet V2 has many layers, so setting the entire model's trainable flag to False will freeze all the layers.
---------------------------------------------------------------------------------------------------------------
'''

base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Add a classification head                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To generate predictions from the block of features, average over the spatial 5x5 spatial locations, 
using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.
---------------------------------------------------------------------------------------------------------------
'''
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print('feature_batch_average.shape = \n{0}\n'.format(feature_batch_average.shape))

'''
---------------------------------------------------------------------------------------------------------------
Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. 
You don't need an activation function here because this prediction will be treated as a logit, 
or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.
---------------------------------------------------------------------------------------------------------------
'''
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print('prediction_batch.shape = \n{0}\n'.format(prediction_batch.shape))

'''
---------------------------------------------------------------------------------------------------------------
Now stack the feature extractor, and these two layers using a tf.keras.Sequential model:
---------------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential([
			base_model,
			global_average_layer,
			prediction_layer
		])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Compile the model                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
You must compile the model before training it. Since there are two classes, use a binary cross-entropy loss.
---------------------------------------------------------------------------------------------------------------
'''
base_learning_rate = 0.0001
model.compile(
				optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
				loss='binary_crossentropy',
				metrics=['accuracy']
			)

model.summary()

'''
---------------------------------------------------------------------------------------------------------------
The 2.5M parameters in MobileNet are frozen, 
but there are 1.2K trainable parameters in the Dense layer. 
These are divided between two tf.Variable objects, the weights and biases.
---------------------------------------------------------------------------------------------------------------
'''
print('len(model.trainable_variables) = \n{0}\n'.format(len(model.trainable_variables)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
After training for 10 epochs, you should see ~96% accuracy.
--------------------------------------------------------------------------------------------------------------
'''
num_train, num_val, num_test = (
									metadata.splits['train'].num_examples*weight/10
									for weight in SPLIT_WEIGHTS
								)

initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(
						train_batches,
						epochs=initial_epochs,
						validation_data=validation_batches
                    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Learning curves                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Let's take a look at the learning curves of the training 
and validation accuracy/loss when using the MobileNet V2 base model as a fixed feature extractor.
--------------------------------------------------------------------------------------------------------------
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

'''
-------------------------------------------------------------------------------------------------------------
Note: 
If you are wondering why the validation metrics are clearly better than the training metrics, 
the main factor is because layers like tf.keras.layers.BatchNormalization and tf.keras.layers.Dropout affect accuracy during training. 
They are turned off when calculating validation loss.

To a lesser extent, it is also because training metrics report the average for an epoch, 
while validation metrics are evaluated after the epoch, so validation metrics see a model that has trained slightly longer.
-------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Fine tuning                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
In our feature extraction experiment, you were only training a few layers on top of an MobileNet V2 base model. 
The weights of the pre-trained network were not updated during training.

One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. 
The training process will force the weights to be tuned from generic features maps to features associated specifically to our dataset.

Note: 
This should only be attempted after you have trained the top-level classifier with the pre-trained model set to non-trainable. 
If you add a randomly initialized classifier on top of a pre-trained model and attempt to train all layers jointly, 
the magnitude of the gradient updates will be too large (due to the random weights from the classifier) and your pre-trained model will forget what it has learned.

Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. 
In most convolutional networks, the higher up a layer is, the more specialized it is. 
The first few layers learn very simple and generic features which generalize to almost all types of images. As you go higher up, 
the features are increasingly more specific to the dataset on which the model was trained. 
The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Un-freeze the top layers of the model                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
All you need to do is unfreeze the base_model and set the bottom layers be un-trainable. 
Then, you should recompile the model (necessary for these changes to take effect), and resume training.
---------------------------------------------------------------------------------------------------------------
'''
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
	layer.trainable =  False

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Compile the model                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Compile the model using a much lower training rate.
model.compile(
				loss='binary_crossentropy',
				optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
				metrics=['accuracy']
			)

model.summary()

print('len(model.trainable_variables) = {0}\n'.format(len(model.trainable_variables)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Continue Train the model                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
If you trained to convergence earlier, this will get you a few percent more accuracy.
---------------------------------------------------------------------------------------------------------------
'''
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
							train_batches,
							epochs=total_epochs,
							initial_epoch =  history.epoch[-1],
							validation_data=validation_batches
				)

'''
---------------------------------------------------------------------------------------------------------------
Let's take a look at the learning curves of the training and validation accuracy/loss, 
when fine tuning the last few layers of the MobileNet V2 base model and training the classifier on top of it. 
The validation loss is much higher than the training loss, so you may get some overfitting.

You may also get some overfitting as the new training set is relatively small and similar to the original MobileNet V2 datasets.

After fine tuning the model nearly reaches 98% accuracy.
--------------------------------------------------------------------------------------------------------------
'''

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Summary:                                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
* Using a pre-trained model for feature extraction: 
When working with a small dataset, it is common to take advantage of features learned by a model trained on a larger dataset in the same domain. 
This is done by instantiating the pre-trained model and adding a fully-connected classifier on top. 
The pre-trained model is "frozen" and only the weights of the classifier get updated during training. 
In this case, the convolutional base extracted all the features associated with each image 
and you just trained a classifier that determines the image class given that set of extracted features.

* Fine-tuning a pre-trained model: 
To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning. 
In this case, you tuned your weights such that your model learned high-level features specific to the dataset. 
This technique is usually recommended when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        images_transfer_learning.py                  　                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()