# common library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Creating data set
X, Y, coef = make_regression(random_state=12, 
                       n_samples=100, 
                       n_features=4,
                       n_informative=2,
                       noise=10.0,
                       bias=-0.0,
                       coef=True)

print("X =", X[:5])
print("Y =", Y[:5])
print("coef =", coef)

plt.figure(figsize=(20, 4))
plt.subplot(1, 4, 1)
plt.title("Feature 1")
plt.plot(X[:, 0], Y, "bo")
plt.subplot(1, 4, 2)
plt.title("Feature 2")
plt.plot(X[:, 1], Y, "ro")
plt.subplot(1, 4, 3)
plt.title("Feature 3")
plt.plot(X[:, 2], Y, "go")
plt.subplot(1, 4, 4)
plt.title("Feature 4")
plt.plot(X[:, 3], Y, "yo")
plt.show()


