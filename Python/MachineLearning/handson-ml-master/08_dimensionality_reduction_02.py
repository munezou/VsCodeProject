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
from sklearn.datasets import make_moons

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN


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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Using clustering for image segmentation                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

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

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '          Using Clustering for Preprocessing                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# Let's tackle the digits dataset which is a simple MNIST-like dataset containing 1,797 grayscale 8×8 images representing digits 0 to 9.
X_digits, y_digits = load_digits(return_X_y=True)

# Let's split it into a training set and a test set: test_size(default)=0.25
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

# Now let's fit a Logistic Regression model and evaluate it on the test set:
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg_fit = log_reg.fit(X_train, y_train)

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------------
Okay, that's our baseline: 96.7% accuracy. 
Let's see if we can do better by using K-Means as a preprocessing step. 
We will create a pipeline that will first cluster the training set into 50 clusters 
and replace the images with their distances to the 50 clusters, then apply a logistic regression model:
--------------------------------------------------------------------------------------------------------------
'''
pipeline = Pipeline([
                    ("kmeans", KMeans(n_clusters=50, random_state=42)),
                    ("log_reg", LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)),
                    ])

pipeline_fit = pipeline.fit(X_train, y_train)
print('pipeline_fit = \n{0}\n'.format(pipeline_fit))

pipeline_score = pipeline.score(X_test, y_test)
print('pipeline_score = {0}\n'.format(pipeline_score))

error_rate = 1 - (1 - 0.9822222) / (1 - 0.9666666)
print('error_rate = {0}\n'.format(error_rate))

'''
-------------------------------------------------------------------------------------------------------------
 How about that? 
 We almost divided the error rate by a factor of 2!
 But we chose the number of clusters  𝑘  completely arbitrarily, we can surely do better. 
 Since K-Means is just a preprocessing step in a classification pipeline, 
finding a good value for  𝑘  is much simpler than earlier: 

 there's no need to perform silhouette analysis or minimize the inertia, 
the best value of  𝑘  is simply the one that results in the best classification performance.
-------------------------------------------------------------------------------------------------------------
'''
param_grid = dict(kmeans__n_clusters=range(2, 100))

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)

print()
print()
grid_clf_fit = grid_clf.fit(X_train, y_train)
print('grid_clf = \n{0}\n'.format(grid_clf_fit))

grid_clf_best_parametrer = grid_clf.best_params_
print('grid_clf_best_parametrer = {0}\n'.format(grid_clf_best_parametrer))

grid_clf_score = grid_clf.score(X_test, y_test)
print('grid_clf_score = {0}\n'.format(grid_clf_score))

'''
-------------------------------------------------------------------------------------------------------------
The performance is slightly improved when  𝑘=90 , so 90 it is.
-------------------------------------------------------------------------------------------------------------
'''
print   ('------------------------------------------------------------------------------------------------------\n'
        '          Clustering for Semi-supervised Learning                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------------------------
Another use case for clustering is in semi-supervised learning, 
when we have plenty of unlabeled instances and very few labeled instances.

Let's look at the performance of a logistic regression model when we only have 50 labeled instances:
------------------------------------------------------------------------------------------------------------
'''
# a number of label specified is 50.
n_labeled = 50

# the used regression is a logistic Regression.
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

# Use logistic regression to fit the data.
log_reg_fit = log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------------
It's much less than earlier of course. Let's see how we can do better. 
First, let's cluster the training set into 50 clusters, then for each cluster let's find the image closest to the centroid.
We will call these images the representative images:
--------------------------------------------------------------------------------------------------------------
'''

k = 50

kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# Now let's plot these representative images and label them manually:
plt.figure(figsize=(8, 2))

for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

save_fig("representative_images_diagram", tight_layout=False)
plt.show()

y_representative_digits = np.array  ([
                                    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
                                    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
                                    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
                                    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
                                    4, 2, 9, 4, 7, 6, 2, 3, 1, 1
                                    ])

'''
-------------------------------------------------------------------------------------------------------
Now we have a dataset with just 50 labeled instances, 
but instead of being completely random instances, each of them is a representative image of its cluster. 
Let's see if the performance is any better:
-------------------------------------------------------------------------------------------------------
'''
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg.fit(X_representative_digits, y_representative_digits)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
-------------------------------------------------------------------------------------------------------
Wow! 
We jumped from 82.7% accuracy to 92.4%, although we are still only training the model on 50 instances. 
Since it's often costly and painful to label instances, especially when it has to be done manually by experts, 
it's a good idea to make them label representative instances rather than just random instances.

But perhaps we can go one step further: 
what if we propagated the labels to all the other instances in the same cluster?
-------------------------------------------------------------------------------------------------------
'''
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

print('y_train_propagated = \n{0}\n'.format(y_train_propagated))

log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg_fit = log_reg.fit(X_train, y_train_propagated)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------------------
We got a tiny little accuracy boost. 
Better than nothing, but we should probably have propagated the labels only to the instances closest to the centroid, 
because by propagating to the full cluster, we have certainly included some outliers. 
Let's only propagate the labels to the 20th percentile closest to the centroid:
--------------------------------------------------------------------------------------------------------
'''
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)

log_reg_fit = log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print('log_reg_fit = \n{0}\n'.format(log_reg_fit))

log_reg_score = log_reg.score(X_test, y_test)
print('log_reg_score = {0}\n'.format(log_reg_score))

'''
--------------------------------------------------------------------------------------------
Nice! With just 50 labeled instances (just 5 examples per class on average!),
we got 94.2% performance, 
which is pretty close to the performance of logistic regression on the fully labeled digits dataset (which was 96.7%).

This is because the propagated labels are actually pretty good: their accuracy is very close to 99%:
--------------------------------------------------------------------------------------------
'''
accuracy = np.mean(y_train_partially_propagated == y_train[partially_propagated])
print('accuracy = {0}\n'.format(accuracy))

'''
---------------------------------------------------------------------------------------------
You could now do a few iterations of active learning:

1.Manually label the instances that the classifier is least sure about, if possible by picking them in distinct clusters.
2.Train a new model with these additional labels.
---------------------------------------------------------------------------------------------
'''
print   ('------------------------------------------------------------------------------------------------------\n'
        '          DBSCAN                                                                                       \n'
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

# used clustering is DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=5)

dbscan_fit = dbscan.fit(X)
print('dbscan_fit = \n{0}\n'.format(dbscan_fit))

print('dbscan.labels_[:10] = \n{0}\n'.format(dbscan.labels_[:10]))

print('dbscan.components_[:3] = \n{0}\n'.format(dbscan.components_[:3]))

dbscan_unique_label = np.unique(dbscan.labels_)
print('dbscan_unique_label = \n{0}\n'.format(dbscan_unique_label))

dbscan2 = DBSCAN(eps=0.2)

dbscan2_fit = dbscan2.fit(X)
print('dbscan2_fit = \n{0}\n'.format(dbscan2_fit))

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

# to compare eps=0.05 and eps=0.20 in DBSCAN
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

save_fig("dbscan_diagram")
plt.show()

dbscan = dbscan2

# 
knn = KNeighborsClassifier(n_neighbors=50)

knn_fit = knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
print('knn_fit = \n{0}\n'.format(knn_fit))

# create Data for verification
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])

# confirm prediction data
knn_predict = knn.predict(X_new)
print('knn_predict = \n{0}\n'.format(knn_predict))

# Calculate probability estimates for X_new.
knn_predict_proba = knn.predict_proba(X_new)
print('knn_predect_proba = \n{0}\n'.format(knn_predict_proba))

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# plot knn result
plt.figure(figsize=(6, 3))
plot_decision_boundaries(knn, X, show_centroids=False)
plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
save_fig("cluster_classification_diagram")
plt.show()

'''
----------------------------------------------------------------------------------------
Finds the K-neighbors of a point. 
Returns indices of and distances to the neighbors of each point.
----------------------------------------------------------------------------------------
'''
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)

y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1

print('y_pred.ravel() = \n{0}\n'.format(y_pred.ravel()))
