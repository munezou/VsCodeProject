# common library
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print('-------------------------------------------------------------------------------------------------------------------\n'
      ' 8.                                                                                                                \n'
      ' Exercise:                                                                                                         \n'
      ' train a LinearSVC on a linearly separable dataset.                                                                \n'
      ' Then train an SVC and a SGDClassifier on the same dataset.                                                        \n'
      ' See if you can get them to produce roughly the same model.                                                        \n'
      '                                                                                                                   \n'
      ' Let us use the Iris dataset: the Iris Setosa and Iris Versicolor classes are linearly separable.                  \n'
      '-------------------------------------------------------------------------------------------------------------------\n')

# Read iris dataset
iris = datasets.load_iris()

# check iris dataset
print('iris dataset information = \n{0}\n'.format(iris["DESCR"]))

# prepare learning data
X = iris["data"][:, (2, 3)]
y = iris["target"]

# plot data condition
plt.figure(figsize=(8, 7))
plt.title("iris raw data related by petal")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c = 'red', label= 'Setosa')
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c = 'blue', label = 'Versicolour')
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], c = 'lime', label = 'Virginica')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.grid(True)
plt.legend()
plt.show()

# Extract target = 0 (Setosa) and target = 1 (Versicolour).
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# Classification of setosa and versicolour
plt.figure(figsize=(8, 7))
plt.title("setosa and versicolour data")
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c = 'red', label= 'Setosa')
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c = 'blue', label = 'Versicolour')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.grid(True)
plt.legend()
plt.show()

# Compare LinearSVC, SVC and SGDClassifier.
C = 5
alpha = 1 / (C * len(X))

lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
svm_clf = SVC(kernel="linear", C=C)
sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha, max_iter=100000, tol=-np.infty, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lin_clf.fit(X_scaled, y)
svm_clf.fit(X_scaled, y)
sgd_clf.fit(X_scaled, y)

print('LinearSVC:\nlin_clf.intercept_ = {0}\nlin_clf.coef_ = {1}\n'.format(lin_clf.intercept_, lin_clf.coef_))
print('SVC:\nsvm_clf.intercept_ = {0}\nsvm_clf.coef_ = {1}\n'.format(svm_clf.intercept_, svm_clf.coef_))
print('SGDClassifier(alpha={0:.5f}):\nsgd_clf.intercept_ = {1}\nsgd_clf.coef_\n'.format(sgd_clf.alpha, sgd_clf.intercept_, sgd_clf.coef_))
print()

print('Let us plot the decision boundaries of these three models:')
# Compute the slope and bias of each decision boundary
w1 = -lin_clf.coef_[0, 0]/lin_clf.coef_[0, 1]
b1 = -lin_clf.intercept_[0]/lin_clf.coef_[0, 1]
w2 = -svm_clf.coef_[0, 0]/svm_clf.coef_[0, 1]
b2 = -svm_clf.intercept_[0]/svm_clf.coef_[0, 1]
w3 = -sgd_clf.coef_[0, 0]/sgd_clf.coef_[0, 1]
b3 = -sgd_clf.intercept_[0]/sgd_clf.coef_[0, 1]

# Transform the decision boundary lines back to the original scale
line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])
line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])
line3 = scaler.inverse_transform([[-10, -10 * w3 + b3], [10, 10 * w3 + b3]])

# Plot all three decision boundaries
plt.figure(figsize=(11, 4))
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs") # label="Iris-Versicolor"
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo") # label="Iris-Setosa"
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper center", fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.show()
