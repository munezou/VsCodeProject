'''
------------------------------------------------------------------------------------------
genarative
    CycleGAN

This notebook demonstrates unpaired image to image translation using conditional GAN's, as described in Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, also known as CycleGAN. 
The paper proposes a method that can capture the characteristics of one image domain and figure out how these characteristics could be translated into another image domain, all in the absence of any paired training examples.

This notebook assumes you are familiar with Pix2Pix, which you can learn about in the Pix2Pix tutorial. 
The code for CycleGAN is similar, the main difference is an additional loss function, and the use of unpaired training data.

CycleGAN uses a cycle consistency loss to enable training without the need for paired data. 
In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain.

This opens up the possibility to do a lot of interesting tasks like photo-enhancement, image colorization, style transfer, etc. 
All you need is the source and the target dataset (which is simply a directory of images).

Output Image 1 Output Image 2
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
import tempfile
import functools
import time
import glob
import imageio
import datetime
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
PROJECT_ROOT_DIR = basic_path.joinpath('Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)



print   (
        '---------------------------------------------------------------------------------\n'
        '      What are GANs?                                                             \n'
        '---------------------------------------------------------------------------------\n'
        )

data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print   (
        '       finished        generative_CycleGan.py                 ({0})                \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()