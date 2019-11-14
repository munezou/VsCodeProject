# common library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Classification data set generation
X, Y = make_blobs(random_state=8,
                  n_samples=100, 
                  n_features=2, 
                  cluster_std=1.5,
                  centers=3)

print("X =", X[:3])
print("Y =", Y)

plt.figure(figsize=(8, 7))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()