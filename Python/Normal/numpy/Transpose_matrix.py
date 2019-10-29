import numpy as np

matrix = [
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34]
]

# Extract matrix elements
tr = []

for vector in matrix:
    tr.append(vector[0])

print('tr = {0}'.format(tr))
print()

print('----------------------------------------------------\n'
     '               Extract matrix elements by normal     \n'
     '-----------------------------------------------------\n')
tr.clear()

for i in range(4):
    tr_row = []
    for vector in matrix:
        tr_row.append(vector[i])
    tr.append(tr_row)

print('tr = \n{0}'.format(tr))
print()

print('----------------------------------------------------\n'
     '               Extract matrix elements by numpy      \n'
     '-----------------------------------------------------\n')
matrix_n = np.array(matrix)

tr.clear()

tr = matrix_n.T

print('tr = \n{0}'.format(tr))
print()
