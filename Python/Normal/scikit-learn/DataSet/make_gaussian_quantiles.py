# common library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

# create data set
X, Y = make_gaussian_quantiles(random_state=42,
                               n_features=2, 
                               n_classes=2)
print("X =", X[:3])
print("Y =", Y[:20])

plt.figure(figsize=(8, 7))
plt.title("make_gaussian_quantiles : n_features=2  n_classes=2")
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()