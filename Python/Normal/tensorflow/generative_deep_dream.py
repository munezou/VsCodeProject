'''
------------------------------------------------------------------------------------------
genarative
    DeepDream

This tutorial contains a minimal implementation of DeepDream, as described in this blog post by Alexander Mordvintsev.

DeepDream is an experiment that visualizes the patterns learned by a neural network. 
Similar to when a child watches clouds and tries to interpret random shapes, 
DeepDream over-interprets and enhances the patterns it sees in an image.

It does so by forwarding an image through the network, 
then calculating the gradient of the image with respect to the activations of a particular layer. 
The image is then modified to increase these activations, 
enhancing the patterns seen by the network, and resulting in a dream-like image. 
This process was dubbed "Inceptionism" (a reference to InceptionNet, and the movie Inception.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
import tempfile
import functools
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
pd.options.display.max_rows = None
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

'''
------------------------------------------------------------------------------------------
Let's demonstrate how you can make a neural network "dream" and enhance the surreal patterns it sees in an image.
------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/dogception.png'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Choose an image to dream-ify                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
For this tutorial, let's use an image of a labrador.
------------------------------------------------------------------------------------------
'''
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# Download an image and read it into a NumPy array.
def download(url, target_size=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    return img

# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


# Display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# Downsizing the image makes it easier to work with.
original_img = download(url, target_size=[225, 375])
original_img = np.array(original_img)

show(original_img)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Prepare the feature extraction model                                       \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Download and prepare a pre-trained image classification model. 
You will use InceptionV3 which is similar to the model originally used in DeepDream. 
Note that any pre-trained model will work, 
although you will have to adjust the layer names below if you change this.
-----------------------------------------------------------------------------------------
'''
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

'''
----------------------------------------------------------------------------------------
The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way 
that the image increasingly "excites" the layers. 
The complexity of the features incorporated depends on layers chosen by you, i.e, 
lower layers produce strokes or simple patterns, while deeper layers give sophisticated features in images, or even whole objects.

The InceptionV3 architecture is quite large 
(for a graph of the model architecture see TensorFlow's research repo). 
For DeepDream, the layers of interest are those where the convolutions are concatenated. 
There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'. 
Using different layers will result in different dream-like images. 
Deeper layers respond to higher-level features (such as eyes and faces), 
while earlier layers respond to simpler features (such as edges, shapes, and textures). 
Feel free to experiment with the layers selected below, 
but keep in mind that deeper layers (those with a higher index) will take longer to train on since the gradient computation is deeper.
-----------------------------------------------------------------------------------------
'''
# Maximize the activations of these layers
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Calculate loss                                                             \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
The loss is the sum of the activations in the chosen layers. 
The loss is normalized at each layer so the contribution from larger layers does not outweigh smaller layers. 
Normally, loss is a quantity you wish to minimize via gradient descent. 
In DeepDream, you will maximize this loss via gradient ascent.
------------------------------------------------------------------------------------------
'''
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Gradient ascent                                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Once you have calculated the loss for the chosen layers, 
all that is left is to calculate the gradients with respect to the image, and add them to the original image.

Adding the gradients to the image enhances the patterns seen by the network. 
At each step, 
you will have created an image that increasingly excites the activations of certain layers in the network.

The method that does this, below, is wrapped in a tf.function for performance. 
It uses an input_signature to ensure that the function is not retraced for different image sizes or steps/step_size values. 
See the Concrete functions guide for details.
-------------------------------------------------------------------------------------------
'''
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8 
            
            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img

deepdream = DeepDream(dream_model)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Main Loop                                                                  \n'
        '---------------------------------------------------------------------------------\n'
        )

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        
        display.clear_output(wait=True)
        show(deprocess(img))
        print ("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)

    return result

dream_img = run_deep_dream_simple(
                img=original_img, 
                steps=100, step_size=0.01
            )

print   (
        '---------------------------------------------------------------------------------\n'
        '      Taking it up an octave                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Pretty good, but there are a few issues with this first attempt:

	1. The output is noisy (this could be addressed with a tf.image.total_variation loss).
	2. The image is low resolution.
	3. The patterns appear like they're all happening at the same granularity.
	
One approach that addresses all these problems is applying gradient ascent at different scales. 
This will allow patterns generated at smaller scales to be incorporated into patterns at higher scales and filled in with additional detail.

To do this you can perform the previous gradient ascent approach, then increase the size of the image (which is referred to as an octave), and repeat this process for multiple octaves.
------------------------------------------------------------------------------------------
'''
start = time.time()

OCTAVE_SCALE = 1.30

img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
float_base_shape = tf.cast(base_shape, tf.float32)

for n in range(-2, 3):
    new_shape = tf.cast(float_base_shape*(OCTAVE_SCALE**n), tf.int32)

    img = tf.image.resize(img, new_shape).numpy()

    img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)

display.clear_output(wait=True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)

end = time.time()
print('end-start = {0}\n'.format(end - start))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Optional: Scaling up with tiles                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
One thing to consider is that as the image increases in size, 
so will the time and memory necessary to perform the gradient calculation. 
The above octave implementation will not work on very large images, or many octaves.

To avoid this issue you can split the image into tiles and compute the gradient for each tile.

Applying random shifts to the image before each tiled computation prevents tile seams from appearing.

Start by implementing the random shift:
-----------------------------------------------------------------------------------------
'''
def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0],shift[1] 
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled

shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
show(img_rolled)

'''
------------------------------------------------------------------------------------------
Here is a tiled equivalent of the deepdream function defined earlier:
------------------------------------------------------------------------------------------
'''
class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    
    def __call__(self, img, tile_size=512):
        shift_down, shift_right, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)
        
        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        return gradients

get_tiled_gradients = TiledGradients(dream_model)

'''
------------------------------------------------------------------------------
Putting this together gives a scalable, octave-aware deepdream implementation:
------------------------------------------------------------------------------
'''
def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, 
                                octaves=range(-2,3), octave_scale=1.3):
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print ("Octave {}, Step {}".format(octave, step))
        
    result = deprocess(img)
    return result

img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)

display.clear_output(wait=True)
img = tf.image.resize(img, base_shape)
img = tf.image.convert_image_dtype(img/255.0, dtype=tf.uint8)
show(img)

'''
---------------------------------------------------------------------------------------
Much better! Play around with the number of octaves, octave scale, 
and activated layers to change how your DeepDream-ed image looks.

Readers might also be interested in TensorFlow Lucid which expands on ideas introduced 
in this tutorial to visualize and interpret neural networks.
---------------------------------------------------------------------------------------
'''
