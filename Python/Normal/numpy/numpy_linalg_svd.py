# common library
import numpy as np
import matplotlib.pyplot as plt

# prepare data
a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

# A: Complex number
Xa = a.real
Ya = a.imag

# B: Complex number
Xb = b.real
Yb = b.imag

plt.figure(figsize=(8, 6))
plt.title(" vector A and B")
plt.scatter(Xa, Ya, c='red', label="A")
plt.scatter(Xb, Yb, c='blue', label="B")
plt.xlabel("real")
plt.ylabel("imag")
plt.grid(True)
plt.show()

# Reconstruction based on full SVD, 2D case:a (rows x columns)
print('a Array = \n{0}\n'.format(a))

u, s, vh = np.linalg.svd(a, full_matrices=False)
print('u.shape = {0}, s.shape = {1}, vh.shape = {2}\n'.format( u.shape, s.shape, vh.shape))

# check unitary matrix(rows x rows)
print('u = \n{0}\n'.format(u))

# check Real diagonal matrix(rows x columns) with the singular values
print('s = \n{0}\n'.format(s))

# check Unitary array(columns x columns)
print('vh = \n{0}\n'.format(vh))

# check Real diagonal matrix
real_diagonal_matrix = np.diag(s)
print('real_diagonal_matrix = \n{0}\n'.format(real_diagonal_matrix))

# Check if the matrix returns to the original A matrix by inverse matrix transformation.
a_transfer = np.dot(np.dot(u, np.diag(s)), vh)
print('a_transfer = \n{0}\n'.format(a_transfer))

# confirm transfer error
error = a - a_transfer
print('error = \n{0}\n'.format(error))

# confirm u(Unitary matrix)
unitary_u = np.dot(u, u.T)
print('unitary_u = \n{0}\n'.format(unitary_u))

# confirm vh(Unitary matrix)
unitary_vh = np.dot(vh, vh.T)
print('unitary_vh = \n{0}\n'.format(unitary_vh))
