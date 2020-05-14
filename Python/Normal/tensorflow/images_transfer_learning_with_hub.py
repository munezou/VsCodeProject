'''
------------------------------------------------------------------------------------------
images
    Transfer learning with TensorFlow Hub

TensorFlow Hub is a way to share pretrained model components. 
See the TensorFlow Module Hub for a searchable listing of pre-trained models. 
This tutorial demonstrates:

	1. How to use TensorFlow Hub with tf.keras.
	2. How to do image classification using TensorFlow Hub.
	3. How to do simple transfer learning.
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

from tensorflow.keras import layers
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
        '       An ImageNet classifier                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the classifier                                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Use hub.module to load a mobilenet, and tf.keras.layers.Lambda to wrap it up as a keras layer. 
Any TensorFlow 2 compatible image classifier URL from tfhub.dev will work here.
---------------------------------------------------------------------------------------------------------------
'''
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
                hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
            ])

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Run it on a single image                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Download a single image to try the model on.
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper.show()

grace_hopper = np.array(grace_hopper)/255.0

print('grace_hopper.shape = {0}\n'.format(grace_hopper.shape))

# Add a batch dimension, and pass the image to the model.
result = classifier.predict(grace_hopper[np.newaxis, ...])

print('result.shape = {0}\n'.format(result.shape))

'''
-------------------------------------------------------------------------------------------------------------
The result is a 1001 element vector of logits, rating the probability of each class for the image.

So the top class ID can be found with argmax:
-------------------------------------------------------------------------------------------------------------
'''
predicted_class = np.argmax(result[0], axis=-1)

print('predicted_class = {0}\n'.format(predicted_class))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Decode the predictions                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
We have the predicted class ID, Fetch the ImageNet labels, and decode the predictions
---------------------------------------------------------------------------------------------------------------
'''
labels_path = tf.keras.utils.get_file(
                    'ImageNetLabels.txt',
                    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
                )
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Simple transfer learning                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Using TF Hub it is simple to retrain the top layer of the model to recognize the classes in our dataset.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Dataset                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
For this example you will use the TensorFlow flowers dataset:
--------------------------------------------------------------------------------------------------------------
'''

data_root = tf.keras.utils.get_file(
                origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                fname=PROJECT_ROOT_DIR.joinpath('original_data', 'flower_photos'), 
                untar=True,
                cache_dir=PROJECT_ROOT_DIR.joinpath('original_data')
            )

'''
--------------------------------------------------------------------------------------------------------------
The simplest way to load this data into our model is using tf.keras.preprocessing.image.ImageDataGenerator,

All of TensorFlow Hub's image modules expect float inputs in the [0, 1] range. 
Use the ImageDataGenerator's rescale parameter to achieve this.

The image size will be handled later.
-------------------------------------------------------------------------------------------------------------
'''
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

'''
------------------------------------------------------------------------------------------------------------
The resulting object is an iterator that returns image_batch, label_batch pairs.
------------------------------------------------------------------------------------------------------------
'''
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Run the classifier on a batch of images                                                        \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
Now run the classifier on the image batch.
--------------------------------------------------------------------------------------------------------------
'''
result_batch = classifier.predict(image_batch)

print('result_batch.shape = \n{0}\n'.format(result_batch.shape))

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

print('predicted_class_names = {0}\n'.format(predicted_class_names))

'''
---------------------------------------------------------------------------------------------------------------
Now check how these predictions line up with the images:
---------------------------------------------------------------------------------------------------------------
'''
# Now check how these predictions line up with the images:
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')

_ = plt.suptitle("ImageNet predictions")
plt.show()

'''
-----------------------------------------------------------------------------------------------------------
See the LICENSE.txt file for image attributions.

The results are far from perfect, 
but reasonable considering that these are not the classes the model was trained for (except "daisy").
----------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Download the headless model                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------
TensorFlow Hub also distributes models without the top classification layer. 
These can be used to easily do transfer learning.

Any Tensorflow 2 compatible image feature vector URL from tfhub.dev will work here.
----------------------------------------------------------------------------------------------------------
'''
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

# Create the feature extractor.
feature_extractor_layer = hub.KerasLayer(
                            feature_extractor_url,
                            input_shape=(224,224,3)
                        )

'''
---------------------------------------------------------------------------------------------------------
It returns a 1280-length vector for each image:
---------------------------------------------------------------------------------------------------------
'''
feature_batch = feature_extractor_layer(image_batch)

print('feature_batch.shape - {0}\n'.format(feature_batch.shape))

'''
--------------------------------------------------------------------------------------------------------
Freeze the variables in the feature extractor layer, 
so that the training only modifies the new classifier layer.
--------------------------------------------------------------------------------------------------------
'''
feature_extractor_layer.trainable = False

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Attach a classification head                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------
Now wrap the hub layer in a tf.keras.Sequential model, and add a new classification layer.
--------------------------------------------------------------------------------------------------------
'''
model = tf.keras.Sequential([
            feature_extractor_layer,
            layers.Dense(image_data.num_classes, activation='softmax')
        ])

model.summary()

predictions = model(image_batch)

print('predictions.shape - \n{0}\n'.format(predictions.shape))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Use compile to configure the training process:
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc']
)

'''
---------------------------------------------------------------------------------------------------------------
Now use the .fit method to train the model.

To keep this example short train just 2 epochs. 
To visualize the training progress, 
use a custom callback to log the loss and accuracy of each batch individually, 
instead of the epoch average.
--------------------------------------------------------------------------------------------------------------
'''
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(
            image_data, epochs=2,
            steps_per_epoch=steps_per_epoch,
            callbacks = [batch_stats_callback]
        )

'''
-------------------------------------------------------------------------------------------------------------
Now after, even just a few training iterations, 
we can already see that the model is making progress on the task.
-------------------------------------------------------------------------------------------------------------
'''
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
plt.show()

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Check the predictions                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To redo the plot from before, first get the ordered list of class names:
---------------------------------------------------------------------------------------------------------------
'''
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])

print('class_names = {0}\n'.format(class_names))

'''
--------------------------------------------------------------------------------------------------------------
Run the image batch through the model and convert the indices to class names.
--------------------------------------------------------------------------------------------------------------
'''
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

# Plot the result
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')

_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Export your model                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Now that you've trained the model, export it as a saved model:
---------------------------------------------------------------------------------------------------------------
'''
t = str(time.time())
file_name = PROJECT_ROOT_DIR.joinpath('tmp', 'saved_models')

export_path = str(file_name.joinpath(t))
model.save(export_path, save_format='tf')

print('export_path = \n{0}\n'.format(export_path))

'''
--------------------------------------------------------------------------------------------------------------
Now confirm that we can reload it, and it still gives the same results:
--------------------------------------------------------------------------------------------------------------
'''
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()

'''
----------------------------------------------------------------------------------------------------------------
This saved model can loaded for inference later, or converted to TFLite or TFjs.
----------------------------------------------------------------------------------------------------------------
'''