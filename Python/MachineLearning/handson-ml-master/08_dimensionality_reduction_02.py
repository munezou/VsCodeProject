# common liblary
import time
import timeit
import numpy as np
import os
import warnings


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.image import imread

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples



# to make this notebook's output stable across runs
np.random.seed(42)

'''
------------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------------
'''
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
CHAPTER_ID = "unsupervised_learning"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")

print('------------------------------------------------------------------------------------------------------\n'
      '          Using clustering for image segmentation                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')
image = imread(os.path.join(PROJECT_ROOT_DIR, "images","unsupervised_learning","ladybug.png"))
print('image.shape = {0}\n'.format(image.shape))

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

save_fig('image_segmentation_diagram', tight_layout=False)
plt.show()

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          Using Clustering for Preprocessing                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Let's tackle the digits dataset which is a simple MNIST-like dataset containing 1,797 grayscale 8×8 images representing digits 0 to 9.
X_digits, y_digits = load_digits(return_X_y=True)

# Let's split it into a training set and a test set:
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# Now let's fit a Logistic Regression model and evaluate it on the test set:
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg_fit = log_reg.fit(X_train, y_train)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------------------------
Okay, that's our baseline: 96.7% accuracy. 
Let's see if we can do better by using K-Means as a preprocessing step. 
We will create a pipeline that will first cluster the training set into 50 clusters 
and replace the images with their distances to the 50 clusters, then apply a logistic regression model:
--------------------------------------------------------------------------------------------------------------------------
'''
