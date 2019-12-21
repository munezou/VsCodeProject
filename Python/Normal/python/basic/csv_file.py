'''
----------------------------------------------------------------------------------------------
CSV file:

1. Start reading and writing CSV
2. Export CSV file
3. Import CSV file
4. If the character code is unknown – for practitioners
----------------------------------------------------------------------------------------------
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
import itertools
from collections import defaultdict
import csv

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

dir_path = os.path.join(PROJECT_ROOT_DIR, "csv_data", "")

# how to write csv file
file_name = "test.csv"
file_path = os.path.join(dir_path, file_name)

# prepare writting data
list_array = [1, 4, 5, 9, 99, 54]
list_array2D = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]

# Write CSV file using with
with open(file_path, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(list_array)
    writer.writerows(list_array2D)

print('---< Read CSV file using with >---') 
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        print(row)

file_name = "test_m.csv"
file_path = os.path.join(dir_path, file_name)

# prepare writting data
list_array = [54, 99, 9, 5, 4, 1]
list_array2D = [
        [6, 5],
        [4, 3],
        [2, 1]
    ]

print()
print()

# Write CSV file using without
f = open(file_path, 'w')

writer = csv.writer(f, lineterminator='\n')
writer.writerow(list_array)
writer.writerows(list_array2D)

f.close()

print('---< Read CSV file using without >---') 
f = open(file_path, 'r')

reader = csv.reader(f)
header = next(reader)
for row in reader:
    print(row)

f.close()