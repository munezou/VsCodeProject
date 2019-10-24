import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

# prepare x data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create histgramdatetime.
sns.distplot(X)
plt.show()

sns.distplot(y)
plt.show()

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('theta_best = \n{0}'.format(theta_best))
print()

print('---< predict data at x = 0 and at x = 2. >---')
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
print('y_predict = \n{0}'.format(y_predict))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X, y)
ax.plot(X_new, y_predict, "r-")
ax.set_title('y = 3 * x + 4 + noise')
ax.set_xlabel('X')
ax.set_ylabel('y')
plt.show()
print()

