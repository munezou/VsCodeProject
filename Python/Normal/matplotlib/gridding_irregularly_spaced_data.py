'''
--------------------------------------------------------------------------------------------
Matplotlib: gridding irregularly spaced data

Date:	2011-09-07 (last modified), 2006-01-22 (created)

A commonly asked question on the matplotlib mailing lists is "how do I make a contour plot of my irregularly spaced data?". 
The answer is, first you interpolate it to a regular grid. 
As of version 0.98.3, matplotlib provides a griddata function that behaves similarly to the matlab version. 
It performs "natural neighbor interpolation" of irregularly spaced data a regular grid, 
which you can then plot with contour, imshow or pcolor.
---------------------------------------------------------------------------------------------
'''
# common lib
import os, sys
import numpy as np
import numpy.matlib
from numpy.random import uniform, seed
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt



print(__doc__)

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/matplotlib')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))


print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Example 1                                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# make up some randomly distributed data
seed(1234)
npts = 200
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)

# define grid.
xi = np.linspace(-2.1,2.1,100)
yi = np.linspace(-2.1,2.1,100)

# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar

# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('griddata test (%d points)' % npts)
plt.show()

