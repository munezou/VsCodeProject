'''
---------------------------------------------------------------------------------------------------------
Data archiving and compression with tar

contents)
The tar format is one of the file archive formats. 
Originally tar is only a file archive and has no compression function, 
but Python allows you to perform file archive and data compression using the tarfile module.
---------------------------------------------------------------------------------------------------------
'''
# common library
import time
import os
import platform
import shutil
import subprocess
from packaging import version
from PIL import Image
import itertools
from collections import defaultdict
import tarfile

import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt

print(__doc__)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Setup                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

pd.options.display.max_columns = None

pf = platform.system()
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

original_data_path = os.path.join(PROJECT_ROOT_DIR, "original_data", "flower_photos", "")

dir_path = os.path.join(PROJECT_ROOT_DIR, "tar_data")

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       write a data by tar                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
with tarfile.open(os.path.join(dir_path, "test.tar.gz"), 'w:gz') as tf:
    tf.add(original_data_path)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       read a data by tar                                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

with tarfile.open(os.path.join(dir_path, "test.tar.gz"), 'r:gz') as tf:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tf, path=os.path.join(dir_path,"test_data"))