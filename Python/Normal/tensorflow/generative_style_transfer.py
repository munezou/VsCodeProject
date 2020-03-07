'''
------------------------------------------------------------------------------------------
generative
    Neural style transfer

This tutorial uses deep learning to compose one image in the style of another image (ever wish you could paint like Picasso or Van Gogh?). 
This is known as neural style transfer and the technique is outlined in A Neural Algorithm of Artistic Style (Gatys et al.).

Note: 
This tutorial demonstrates the original style-transfer algorithm. 
It optimizes the image content to a particular style. 
Modern approaches train a model to generate the stylized image directly (similar to cyclegan). 
This approach is much faster (up to 1000x). 
A pretrained Arbitrary Image Stylization module is available in TensorFlow Hub, and for TensorFlow Lite.

Neural style transfer is an optimization technique used to take two images—a content image 
and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, 
but “painted” in the style of the style reference image.

This is implemented by optimizing the output image to match the content statistics of the content image 
and the style statistics of the style reference image. 
These statistics are extracted from the images using a convolutional network.

For example, let’s take an image of this dog and Wassily Kandinsky's Composition 7:
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
import time
import functools
from pathlib import Path
from PIL import Image

import tensorflow_hub as hub
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/YellowLabradorLooking_new.jpg'))
im.show()

'''
------------------------------------------------------------------------------------------
Yellow Labrador Looking, from Wikimedia Commons
------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/Vassily_Kandinsky,_1913_-_Composition_7.jpg'))
im.show()

'''
------------------------------------------------------------------------------------------
Now how would it look like if Kandinsky decided 
to paint the picture of this Dog exclusively with this style? Something like this?
------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/stylized-image.png'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Import and configure modules                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

'''
-------------------------------------------------------------------------------------------
Download images and choose a style image and a content image:
-------------------------------------------------------------------------------------------
'''
image_path_00 = PROJECT_ROOT_DIR.joinpath('images/YellowLabradorLooking_new.jpg')

content_path = tf.keras.utils.get_file(
                    fname=image_path_00, 
                    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
                )

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
image_path_01 = PROJECT_ROOT_DIR.joinpath('images/kandinsky5.jpg')

style_path = tf.keras.utils.get_file(
                    fname=image_path_01,
                    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
                )

print   (
        '---------------------------------------------------------------------------------\n'
        '      Visualize the input                                                        \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Define a function to load an image and limit its maximum dimension to 512 pixels.
------------------------------------------------------------------------------------------
'''
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

'''
----------------------------------------------------------------------------------------
Create a simple function to display an image:
----------------------------------------------------------------------------------------
'''
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

print   (
        '---------------------------------------------------------------------------------\n'
        '      Fast Style Transfer using TF-Hub                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
This tutorial demonstrates the original style-transfer algorithm. 
Which optimizes the image content to a particular style. 
Before getting into the details let's see how the TensorFlow Hub module does:
-----------------------------------------------------------------------------------------
'''
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Define content and style representations                                   \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Use the intermediate layers of the model to get the content and style representations of the image. 
Starting from the network's input layer, the first few layer activations represent low-level features like edges and textures. 
As you step through the network, the final few layers represent higher-level features—object parts like wheels or eyes. 
In this case, you are using the VGG19 network architecture, a pretrained image classification network. 
These intermediate layers are necessary to define the representation of content and style from the images. 
For an input image, try to match the corresponding style and content target representations at these intermediate layers.

Load a VGG19 and test run it on our image to ensure it's used correctly:
------------------------------------------------------------------------------------------
'''
x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

'''
------------------------------------------------------------------------------------------
Now load a VGG19 without the classification head, and list the layer names
------------------------------------------------------------------------------------------
'''
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

print()
for layer in vgg.layers:
    print(layer.name)

'''
------------------------------------------------------------------------------------------
Choose intermediate layers from the network to represent the style and content of the image:
------------------------------------------------------------------------------------------
'''
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Intermediate layers for style and content                                  \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
So why do these intermediate outputs within our pretrained image classification network allow us 
to define style and content representations?

At a high level, in order for a network to perform image classification 
(which this network has been trained to do), it must understand the image. 
This requires taking the raw image as input pixels and building an internal representation 
that converts the raw image pixels into a complex understanding of the features present within the image.

This is also a reason why convolutional neural networks are able to generalize well: 
they’re able to capture the invariances and defining features within classes 
(e.g. cats vs. dogs) that are agnostic to background noise and other nuisances. 
Thus, somewhere between where the raw image is fed into the model and the output classification label, 
the model serves as a complex feature extractor. 
By accessing intermediate layers of the model, you're able to describe the content and style of input images.
------------------------------------------------------------------------------------------
'''

print   (
        '---------------------------------------------------------------------------------\n'
        '      Build the model                                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
The networks in tf.keras.applications are designed 
so you can easily extract the intermediate layer values using the Keras functional API.

To define a model using the functional API, specify the inputs and outputs:

model = Model(inputs, outputs)

This following function builds a VGG19 model that returns a list of intermediate layer outputs:
-----------------------------------------------------------------------------------------
'''
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

'''
----------------------------------------------------------------------------------------
And to create the model:
----------------------------------------------------------------------------------------
'''
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

