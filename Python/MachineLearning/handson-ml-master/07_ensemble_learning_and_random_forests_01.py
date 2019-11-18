# Common imports
import numpy as np
import os
import time
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_moons
from sklearn.datasets import make_regression
import xgboost

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


'''
--------------------------------------------------------------------
Setup
--------------------------------------------------------------------
'''
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
CHAPTER_ID = "ensembles"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

print('------------------------------------------------------------------------------------------------------\n'
      '          7.4.2 Importance of features                                                                \n'
      '------------------------------------------------------------------------------------------------------\n')
'''
---------------------------------------------------------------------------------------------
a data using iris
----------------------------------------------------------------------------------------------
'''
# load iris data
iris = load_iris()

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

# fitting
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

print()

rnd_clf_feature_importance = rnd_clf.feature_importances_
print('rnd_clf_feature_importance = \n{0}\n'.format(rnd_clf_feature_importance))

'''
---------------------------------------------------------------------------------------------
a data using moons
----------------------------------------------------------------------------------------------
'''
# create moons data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# display raw data of moons.
plt.figure(figsize=(8, 6))
plt.title("raw data of moons(noise=0.3, random_state=42)")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c='red', label="y = 0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='blue', label="y = 1")
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(True)
plt.legend()
plt.show()

print()

# split  data to test data and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.02, contour=False)

plt.show()
print()

'''
---------------------------------------------------------------------------------------------
a data using minist
----------------------------------------------------------------------------------------------
'''
# load mnist data
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int64)

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)

# fitting
rnd_clf_fit = rnd_clf.fit(mnist["data"], mnist["target"])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.hot, interpolation="nearest")
    plt.axis("off")

plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])

save_fig("mnist_feature_importance_plot")
plt.show()
