# common library
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Get iris from the dataset.
iris = datasets.load_iris()

print('list(iris.keys() = \n{0}'.format(list(iris.keys()))) 
print(iris.DESCR)

# classification by pental.
X = iris["data"][:, [2, 3]]
y = (iris["target"]).astype(np.int)

# Split iris data into training data and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )

# learning model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Evaluate the accuracy of the model.
pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
print('Correct answer rate for training data = {0:.2f}'.format(accuracy_train))
print()

# Check for overlearning
pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('Correct answer rate for test data = {0:.2f}'.format(accuracy_test))
print()

plt.style.use('ggplot') 

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

fig = plt.figure(figsize=(13,8))
plot_decision_regions(X_combined, y_combined, clf=model,  res=0.02)
plt.show()