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

p_path = PROJECT_ROOT_DIR.joinpath('Python/Normal/python/basic/txt_data/')

if p_path.exists() == False:
    p_path.mkdir()
else:
    p_path.mkdir(exist_ok=True)
    
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
        '       Delete files and directories, change permissions                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# remove file
p_path.unlink()

p_dir = PROJECT_ROOT_DIR.joinpath('Python/Normal/python/basic/txt_data')

# remove directory
if pf == 'Linux':
    p_dir.chmod(775)
    p_dir.rmdir()
elif pf == 'Windows':
    if p_dir.is_dir() == True:
        shutil.rmtree(p_dir)


print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Path join                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
if pf == 'Linux':
    print('Path(''a/b/c'') / ''d'' = {0}\n'.format(Path('a/b/c') / 'd'))
elif pf == 'Windows':
    print('Path(''a\\b\\c'') / ''d'' = {0}\n'.format(Path('a\\b\\c') / 'd'))

print('---< By method >---')

if pf == 'Linux':
    print('Path(''a/b/c'').joinpath(''d'') = {0}\n'.format(Path('a/b/c').joinpath('d')))
elif pf == 'Windows':
    print('Path(''a\\b\\c'').joinpath(''d'')  = {0}\n'.format(Path('a\\b\\c').joinpath('d')))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       File list acquisition and search                                                               \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

p_code = PROJECT_ROOT_DIR.joinpath('Python/Normal/python/basic')

# Returns a generator of files and directory listings in the path.
print('p_code.iterdir() = {0}\n'.format(p_code.iterdir()))

# Returns the list of files and directories in the path as a list.
all_list = list(p_code.iterdir())
print('all_list = \n{0}\n'.format(all_list))

# Searches for .py files in the path and returns a generator.
print('p_code.glob(''*.py'') = {0}\n'.format(p.glob('*.py')))

# Searches for .py files in the path and returns a list.
py_list = list(p.glob('*.py'))
print('py_list = \n{0}\n'.format(py_list))