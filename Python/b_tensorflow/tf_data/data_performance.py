'''
----------------------------------------------------------------------------------------------
Better performance with the tf.data API

overview)
GPUs and TPUs can radically reduce the time required to execute a single training step. 
Achieving peak performance requires an efficient input pipeline that delivers data for the next step before the current step has finished. 
The tf.data API helps to build flexible and efficient input pipelines. 
This document demonstrates how to use the tf.data API to build highly performant TensorFlow input pipelines.

Before you continue, read the "Build TensorFlow input pipelines" guide, to learn how to use the tf.data API.
---------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import platform
import shutil
import subprocess
from packaging import version
from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf

print(__doc__)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The dataset                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Define a class inheriting from tf.data.Dataset called ArtificialDataset. 
This dataset:

* generates num_samples samples (default is 3)
* sleeps for some time before the first item to simulate opening a file
* sleeps for some time before producing each item to simulate reading data from a file
---------------------------------------------------------------------------------------------------------------
'''
class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)
        
        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)
            
            yield (sample_idx,)
    
    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )

'''
----------------------------------------------------------------------------------------------------------------
This dataset is similar to the tf.data.Dataset.range one, 
adding a fixed delay at the beginning and between each sample.
----------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The training loop                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Write a dummy training loop that measures how long it takes to iterate over a dataset. 
Training time is simulated.
---------------------------------------------------------------------------------------------------------------
'''
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Optimize performance                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
To exhibit how performance can be optimized, you will improve the performance of the ArtificialDataset.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The naive approach                                                                             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
