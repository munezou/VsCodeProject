'''
------------------------------------------------------------------------------------------
scipy.interpolate.griddata

scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)
Interpolate unstructured D-dimensional data.

Parameters:
    * points2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.

    * valuesndarray of float or complex, shape (n,)
        Data values.

    * xi2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.

    * method{‘linear’, ‘nearest’, ‘cubic’}, 
        optional Method of interpolation. 
        
        One of

        1. nearest
        return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.

        2. linear
        tessellate the input point set to n-dimensional simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.

        3. cubic (1-D)
        return the value determined from a cubic spline.

        4. cubic (2-D)
        return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

    * fill_valuefloat, optional
        Value used to fill in for requested points outside of the convex hull of the input points. 
        If not provided, then the default is nan. This option has no effect for the ‘nearest’ method.

    * rescalebool, optional
        Rescale points to unit cube before performing interpolation. 
        This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.

        New in version 0.14.0.

Returns
    * ndarray
        Array of interpolated values.
---------------------------------------------------------------------------------------------------------------------------
'''
# common library
import os, sys
from pathlib import Path
import numpy as np

print(__doc__)

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/matplotlib')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))


print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Examples                                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# Suppose we want to interpolate the 2-D function
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

# on a grid in [0, 1]x[0, 1]
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# but we only know its values at 1000 data points:
points = np.random.rand(1000, 2)

values = func(points[:,0], points[:,1])



