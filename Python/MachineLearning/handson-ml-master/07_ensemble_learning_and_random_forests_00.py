# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score


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


'''
----------------------------------------------------------------------
Chapter 7: Ensemble learning and random forest
----------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          7.1 Voting classifier                                                                       \n'
      '------------------------------------------------------------------------------------------------------\n')
heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
save_fig("law_of_large_numbers_plot")
plt.show()

'''
-------------------------------------------------------------------------------
Create and train a voting classifier composed of three different classifiers.
-------------------------------------------------------------------------------
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

print('---< setting classifier(voting="hard") >---')
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

# fitting
volt_clf_fit = voting_clf.fit(X_train, y_train)
print('volt_clf_fit = \n{0}\n'.format(volt_clf_fit))

# Performance evaluation
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print()

print('---< setting classifier(voting="soft") >---')
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')

# fitting
volt_clf_fit = voting_clf.fit(X_train, y_train)
print('volt_clf_fit = \n{0}\n'.format(volt_clf_fit))

# Performance evaluation
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          7.2.1 Bagging and pasting in scikit-learn                                                   \n'
      '------------------------------------------------------------------------------------------------------\n')
print('---< prepare a Bagging Classifier >---')
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)

# fitting
bag_clf_fit = bag_clf.fit(X_train, y_train)
print('bag_clf_fit = \n{0}\n'.format(bag_clf_fit))

# accuracy result
y_pred = bag_clf.predict(X_test)

# compare target and predict
print('accuracy_score(y_test, y_pred) = {0}\n'.format(accuracy_score(y_test, y_pred)))

print('---< prepare a Decision trees >---')
tree_clf = DecisionTreeClassifier(random_state=42)

# fitting
tree_clf.fit(X_train, y_train)

# accuracy result
y_pred_tree = tree_clf.predict(X_test)

# compare target and predict
print('accuracy_score(y_test, y_pred_tree) = {0}\n'.format(accuracy_score(y_test, y_pred_tree)))

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

plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()

print()

print('------------------------------------------------------------------------------------------------------\n'
      '          7.2.2 Out-of-Bag evaluation                                                                 \n'
      '------------------------------------------------------------------------------------------------------\n')
# Bagging Classifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap_features=True, n_jobs=-1, oob_score=True)

# fitting
bag_clf_fit = bag_clf.fit(X_train, y_train)
print('bag_clf_fit = \n{0}\n'.format(bag_clf_fit))

# OOB verification
bag_clf_score = bag_clf.oob_score_
print('bag_clf_score = {0}\n'.format(bag_clf_score))

# accuracy score
y_pred = bag_clf.predict(X_test)
bad_clf_accuracy = accuracy_score(y_test, y_pred)
print('bad_clf_accuracy = {0}\n'.format(bad_clf_accuracy))

# oob_decision function by X_train
bad_clf_oob_decision = bag_clf.oob_decision_function_
print('bad_clf_oob_decision = \n{0}\n'.format(bad_clf_oob_decision))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.4 Random Forests                                                                          \n'
      '------------------------------------------------------------------------------------------------------\n')
# Bagging Classifier with Decision Trees
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)

# fiting and predict
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# Random Forest Classifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

# fiting and predict
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

# different of Bagging Classifier with Decision and Random Forest Classifier
diff = np.sum(y_pred == y_pred_rf) / len(y_pred)
print('diff = {0}\n'.format(diff))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.4.2 Importance of features                                                                \n'
      '------------------------------------------------------------------------------------------------------\n')

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

plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.02, contour=False)

plt.show()
print()

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

'''
-----------------------------------------------------------------------------------
7.5 Boosting
-----------------------------------------------------------------------------------
'''
print('------------------------------------------------------------------------------------------------------\n'
      '          7.5.1 AdaBoost                                                                              \n'
      '------------------------------------------------------------------------------------------------------\n')
# create moons data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# split  data to test data and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)

# fitting
ada_clf_fit = ada_clf.fit(X_train, y_train)

plot_decision_boundary(ada_clf, X, y)

m = len(X_train)

plt.figure(figsize=(11, 4))
for subplot, learning_rate in ((121, 1), (122, 0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel="rbf", C=0.05, gamma="auto", random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
    if subplot == 121:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)

save_fig("boosting_plot")
plt.show()
print()

list_ada = list(m for m in dir(ada_clf) if not m.startswith("_") and m.endswith("_"))
print('list_ada = \n{0}\n'.format(list_ada))

print('------------------------------------------------------------------------------------------------------\n'
      '          7.5.2 Gradient Boosting                                                                     \n'
      '------------------------------------------------------------------------------------------------------\n')

