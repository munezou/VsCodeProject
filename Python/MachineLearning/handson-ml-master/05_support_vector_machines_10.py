# common library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


print('--------------------------------------------------------------------------------------------------------------\n'
      ' 9.                                                                                                           \n'
      '  train an SVM classifier on the MNIST dataset.                                                               \n'
      '  Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 10 digits.\n'
      '  You may want to tune the hyperparameters using small validation sets to speed up the process.               \n'
      '  What accuracy can you reach?                                                                                \n'
      '  First, let us load the dataset and split it into a training set and a test set.                             \n'
      '  We could use train_test_split().                                                                            \n'
      '  but people usually just take the first 60,000 instances for the training set,                               \n'
      '  and the last 10,000 instances for the test set                                                              \n'
      ' (this makes it possible to compare your model is performance with others):                                   \n'
      '--------------------------------------------------------------------------------------------------------------\n')
mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"]

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# Many training algorithms are sensitive to the order of the training instances, so it's generally good practice to shuffle them first:
np.random.seed(42)
rnd_idx = np.random.permutation(60000)
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

# Let's start simple, with a linear SVM classifier. 
# It will automatically use the One-vs-All (also called One-vs-the-Rest, OvR) strategy, so there's nothing special we need to do. Easy!