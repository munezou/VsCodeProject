'''
------------------------------------------------------------------------------------------
images
    Object detection

This tutorial focuses on the task of image segmentation, using a modified U-Net.

What is image segmentation?

So far you have seen image classification, where the task of the network is to assign a label or class to an input image. 
However, suppose you want to know where an object is located in the image, the shape of that object, which pixel belongs to which object, etc. 
In this case you will want to segment the image, i.e., each pixel of the image is given a label. 
Thus, the task of image segmentation is to train a neural network to output a pixel-wise mask of the image. 
This helps in understanding the image at a much lower level, i.e., the pixel level. 
Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging to name a few.

The dataset that will be used for this tutorial is the Oxford-IIIT Pet Dataset, created by Parkhi et al. The dataset consists of images, 
their corresponding labels, and pixel-wise masks. The masks are basically labels for each pixel. Each pixel is given one of three categories :

	* Class 1 : Pixel belonging to the pet.
	* Class 2 : Pixel bordering the pet.
	* Class 3 : None of the above/ Surrounding pixel.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import pprint
import contextlib
from pathlib import Path
from packaging import version
from PIL import Image
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers
print(__doc__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorflow_examples.models.pix2pix import pix2pix

keras = tf.keras
tfds.disable_progress_bar()

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

pprint.pprint(sys.path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Example use                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Helper functions for downloading images and for visualization.                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Visualization code adapted from TF object detection API for the simplest required functionality.
---------------------------------------------------------------------------------------------------------------
'''
def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_bounding_box_on_image(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        font,
        thickness=4,
        display_str_list=()
    ):

    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
                                    xmin * im_width, 
                                    max * im_width,
                                    ymin * im_height, 
                                    ymax * im_height
                                )

    draw.line(
                [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                width=thickness,
                fill=color
            )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
                fill=color
            )
        
        draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill="black",
                    font=font
                )
        
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                    25
                )
        
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                            int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                    image_pil,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color,
                    font,
                    display_str_list=[display_str]
                )

            np.copyto(image, np.array(image_pil))
    return image

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Apply module                                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Load a public image from Open Images v4, save locally, and display.
---------------------------------------------------------------------------------------------------------------
'''

image_url = "https://farm1.staticflickr.com/4032/4653948754_c0d768086b_o.jpg"  #@param
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)

'''
--------------------------------------------------------------------------------------------------------------
Pick an object detection module and apply on the downloaded image. Modules:

	* FasterRCNN+InceptionResNet V2: high accuracy,
	* ssd+mobilenet V2: small and fast.
--------------------------------------------------------------------------------------------------------------
'''

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]

detector = hub.load(module_handle).signatures['default']

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        images_object_detection.py     　　       　                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()