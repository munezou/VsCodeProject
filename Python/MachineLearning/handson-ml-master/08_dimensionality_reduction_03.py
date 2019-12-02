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

from sklearn.datasets import make_moons

from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering





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

'''
---------------------------------------------------------------------------------------------
Other Clustering Algorithms
---------------------------------------------------------------------------------------------
'''
print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Spectral Clustering                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
# prepare used data by make_moons
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# confirm the data by make_moons
plt.figure(figsize=(8, 6))
plt.title("a raw data by make_moons")
plt.scatter(X[:, 0][y ==0], X[:, 1][y == 0], c='red', label="y = 0")
plt.scatter(X[:, 0][y ==1], X[:, 1][y == 1], c='blue', label="y = 1")
plt.grid(True)
plt.xlabel("X0")
plt.ylabel("X1")
plt.legend()
plt.show()

sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)

sc1_fit = sc1.fit(X)
print('sc1_fit = \n{0}\n'.format(sc1_fit))

sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)

sc2_fit = sc2.fit(X)
print('sc2_fit = \n{0}\n'.format(sc2_fit))

Sc1_affinity_matrix_95 = np.percentile(sc1.affinity_matrix_, 95)
print('Sc1_affinity_matrix_95 = {0}\n'.format(Sc1_affinity_matrix_95))

def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap="Paired", alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")
    
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)

plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

plt.subplot(122)
plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Agglomerative Clustering                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
-------------------------------------------------------------------------------------------------
Agglomerative Clustering

Recursively merges the pair of clusters that minimally increases a given linkage distance.
-------------------------------------------------------------------------------------------------
'''
def learned_parameters(model):
    return [m for m in dir(model)
            if m.endswith("_") and not m.startswith("_")]

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)

agg = AgglomerativeClustering(linkage="complete").fit(X)
print('agg = \n{0}\n'.format(agg))

print('learned_parameters(agg) = \n{0}\n'.format(learned_parameters(agg)))

print('agg.children_ = \n{0}\n'.format(agg.children_))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Gaussian Mixtures                                                                           \n'
        '------------------------------------------------------------------------------------------------------\n'
        )