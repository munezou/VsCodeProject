# common library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# prepare a using data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# display the using data.
plt.figure(figsize=(8, 6))
plt.title("raw data in Voting Classifier method")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c='pink', label="y = 1")
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], c='purple', label="y = 2")
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(True)
plt.legend()
plt.show()

# prepare some classifier
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

print('---< fitting: voting="hard" >---')
eclf1 = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')

# Performance evaluation
for clf in (clf1, clf2, clf3, eclf1):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(clf.__class__.__name__, accuracy_score(y, y_pred))
print()

# prdicted result
print('eclf1.predict(X) = \n{0}\n'.format(eclf1.predict(X)))

# confirm whether a target data is equaled a prdicted data or not.
equal_result = np.array_equal(eclf1.named_estimators_.lr.predict(X), eclf1.named_estimators_['lr'].predict(X))
print('equal_result = {0}\n'.format(equal_result))

print('---< fitting: voting="soft" >---')
eclf2 = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft')

# Performance evaluation
for clf in (clf1, clf2, clf3, eclf2):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(clf.__class__.__name__, accuracy_score(y, y_pred))
print()

# prdicted result
print('eclf2.predict(X) = \n{0}\n'.format(eclf2.predict(X)))

print('---< fitting: voting="soft", weight=[2, 1, 1], flatten_transform=True >---')
eclf3 = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft', weights=[2,1,1],
    flatten_transform=True)

# Performance evaluation
for clf in (clf1, clf2, clf3, eclf3):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(clf.__class__.__name__, accuracy_score(y, y_pred))
print()

# prdicted result
print('eclf3.predict(X) = \n{0}\n'.format(eclf3.predict(X)))

print('eclf3.transform(X).shape = {0}\n'.format(eclf3.transform(X).shape))