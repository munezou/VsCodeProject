import numpy as np
import matplotlib.pyplot as plt
 
plt.figure()
 
# Start point of arrow (vector)
O = np.array([0,0])

# Arrow (vector) component
X = np.array([4,3])
 
# Arrow (vector)
plt.quiver(O[0],O[1],
           X[0],X[1], 
           angles='xy',scale_units='xy',scale=1)
 
# graph display
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.grid()
plt.show()
print()

# numpy.linalg.norm is a function that calculates the norm.
print('np.linalg.norm(X, ord=0) = {0}'.format(np.linalg.norm(X, ord=0)))
print()

print('np.linalg.norm(X, ord=1) = {0}'.format(np.linalg.norm(X, ord=1)))
print()
print('np.sum(np.abs(X)) = {0}'.format(np.sum(np.abs(X))))
print()

print('np.linalg.norm(X, ord=2) = {0}'.format(np.linalg.norm(X, ord=2)))
print()
print('np.sum(np.abs(X**2)) = {0}'.format(np.sum(np.abs(X**2))))
print()
