# common library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# case of two classification
X, Y = make_classification(random_state=12,
                           n_features=2, 
                           n_redundant=0, 
                           n_informative=1,
                           n_clusters_per_class=1,
                           n_classes=2)
print("X =", X[:3])
print("Y =", Y[:20])

plt.figure(figsize=(8, 7))
plt.title("make_classification : n_features=2  n_classes=2")
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# case of three classification
X, Y = make_classification(random_state=11,
                           n_features=2, 
                           n_redundant=0, 
                           n_informative=2,
                           n_clusters_per_class=1, 
                           n_classes=3)
print("X =", X[:3])
print("Y =", Y[:20])

plt.figure(figsize=(8, 7))
plt.title("make_classification : n_features=2  n_classes=3")
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()