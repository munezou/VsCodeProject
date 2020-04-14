'''
------------------------------------------------------------------------
OpenCV
    read image
------------------------------------------------------------------------
'''
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

print(__doc__)

PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = {0}'.format(PROJECT_ROOT_DIR))

messi5_path = str(os.path.join(PROJECT_ROOT_DIR, "images/messi5.jpg"))
print('messi5_path = {0}\n'.format(messi5_path))

# Load an color image in grayscale
img = cv2.imread(messi5_path, 0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
