import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# prepare data
def f(x, y):
    return x**2 + y**2 + x * y

X, Y = np.mgrid[-3:3, -3:3]

print('X:\n', X)
print()
print('Y:\n', Y)
print()

Z = f(X, Y)
print('Z:\n', Z)
print()

fig = plt.figure(figsize=(9, 4))

# Create contour lines.
ax1 = fig.add_subplot(121)
ax1.set_title('contour')
contour = ax1.contourf(X, Y, Z)
print(type(contour))  # <class 'matplotlib.contour.QuadContourSet'>

# Create a 3D graph.
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('surface')
ax2.plot_surface(X, Y, Z)

plt.show()