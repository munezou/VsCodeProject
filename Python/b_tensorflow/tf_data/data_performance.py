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
# Start with a naive pipeline using no tricks, iterating over the dataset as-is.
benchmark(ArtificialDataset())

'''
---------------------------------------------------------------------------------------------------------------
Under the hood, this is how your execution time was spent:
'''
im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "naive.png"))
im.show()

'''
You can see that performing a training step involves:

* opening a file if it hasn't been opened yet,
* fetching a data entry from the file,
* using the data for training.

However, in a naive synchronous implementation like here, while your pipeline is fetching the data, 
your model is sitting idle. Conversely, while your model is training, the input pipeline is sitting idle. 
The training step time is thus the sum of all, opening, reading and training time.

The next sections build on this input pipeline, illustrating best practices for designing performant TensorFlow input pipelines.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Prefetching                                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
Prefetching overlaps the preprocessing and model execution of a training step. 
While the model is executing training step s, the input pipeline is reading the data for step s+1. 
Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.

The tf.data API provides the tf.data.Dataset.prefetch transformation. 
It can be used to decouple the time when data is produced from the time when data is consumed. 
In particular, 
the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested. 
The number of elements to prefetch should be equal to (or possibly greater than) the number of batches consumed by a single training step. 
You could either manually tune this value, or set it to tf.data.experimental.AUTOTUNE 
which will prompt the tf.data runtime to tune the value dynamically at runtime.

Note that the prefetch transformation provides benefits any time there is an opportunity to overlap the work of a "producer" with the work of a "consumer."
------------------------------------------------------------------------------------------------------------------
'''
benchmark(
    ArtificialDataset()
    .prefetch(tf.data.experimental.AUTOTUNE)
)

im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "prefetched.png"))
im.show()
'''
--------------------------------------------------------------------------------------------------------------------
This time you can see that while the training step is running for sample 0, 
the input pipeline is reading the data for the sample 1, and so on.
--------------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Parallelizing data extraction                                                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------------------------------
In a real-world setting, the input data may be stored remotely (for example, GCS or HDFS). 
A dataset pipeline that works well when reading data locally might become bottlenecked on I/O when reading data remotely 
because of the following differences between local and remote storage:

* Time-to-first-byte: 
    Reading the first byte of a file from remote storage can take orders of magnitude longer than from local storage.
* Read throughput: 
    While remote storage typically offers large aggregate bandwidth, reading a single file might only be able to utilize a small fraction of this bandwidth.

In addition, once the raw bytes are loaded into memory, it may also be necessary to deserialize and/or decrypt the data (e.g. protobuf), 
which requires additional computation. 
This overhead is present irrespective of whether the data is stored locally or remotely, 
but can be worse in the remote case if data is not prefetched effectively.

To mitigate the impact of the various data extraction overheads, 
the tf.data.Dataset.interleave transformation can be used to parallelize the data loading step, 
interleaving the contents of other datasets (such as data file readers). 
The number of datasets to overlap can be specified by the cycle_length argument, 
while the level of parallelism can be specified by the num_parallel_calls argument. 
Similar to the prefetch transformation, 
the interleave transformation supports tf.data.experimental.AUTOTUNE which will delegate the decision 
about what level of parallelism to use to the tf.data runtime.
------------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Sequential interleave                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# The default arguments of the tf.data.Dataset.interleave transformation make it interleave single samples from two datasets sequentially.
benchmark(
    tf.data.Dataset.range(2)
    .interleave(ArtificialDataset)
)

im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "sequential_interleave.png"))
im.show()

'''
---------------------------------------------------------------------------------------------------------------
This plot allows to exhibit the behavior of the interleave transformation, 
fetching samples alternatively from the two datasets available. 
However, no performance improvement is involved here.
---------------------------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Parallel interleave                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# Now use the num_parallel_calls argument of the interleave transformation. 
# This loads multiple datasets in parallel, reducing the time waiting for the files to be opened.
benchmark(
    tf.data.Dataset.range(2)
    .interleave(ArtificialDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
)

im = Image.open(os.path.join(PROJECT_ROOT_DIR, "image", "parallel_interleave.png"))
im.show()

'''
--------------------------------------------------------------------------------------------------------------
Now, you can see on the plot that the pre-processing steps overlap, reducing the overall time for a single iteration.
--------------------------------------------------------------------------------------------------------------
'''
