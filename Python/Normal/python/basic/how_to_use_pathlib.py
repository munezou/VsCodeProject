'''
----------------------------------------------------------------------------------------------------------
library: pathlib

contents)
Too good standard library called pathlib
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
from pathlib import Path

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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       get current directory                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
PROJECT_ROOT_DIR = Path.cwd()
print('current path = {0}\n'.format(PROJECT_ROOT_DIR))

p_path = PROJECT_ROOT_DIR.joinpath('Python/Normal/python/basic/txt_data/ccc.txt')
print('p = {0}\n'.format(p_path))

s = '''
        If the mode argument of open () is 'r', the file is opened for reading. 
        The default value of the argument mode is 'r', so it is OK if omitted.

        If mode = 'r' specifies a path that does not exist in the first argument, an error (FileNotFoundError exception) occurs.
    '''

with open(p_path, mode='w') as f:
    f.write(s)

with open(p_path, mode='r') as f:
    print(f.read())

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Obtain absolute path                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
p = Path('.')
print('p = {0}\n'.format(p))
print('p(absolute path) = {0}\n'.format(p.resolve()))
print('p(absolute path) = {0}\n'.format(p.absolute()))
print('p.is_absolute() = {0}\n'.format(p.resolve().is_absolute()))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Get file name and extension                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print('---< Get file name Return value is character string >---')
print('p_path.name = {0}\n'.format(p_path.name))

print('---< Get extension Return value is a character string >---')
print('p_path.suffix = {0}\n'.format(p_path.suffix))

print('---< You can get a list like .tar.gz with .suffixes. >---')
print('p_path.suffixes = {0}\n'.format(p_path.suffixes))

print('---< Get the file name without the extension. The return value is a character string >---')
print('p_path.stem = {0}\n'.format(p_path.stem))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Get file name and extension                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )