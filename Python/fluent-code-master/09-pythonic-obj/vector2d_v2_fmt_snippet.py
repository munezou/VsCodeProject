"""
A 2-dimensional vector class
"""

# common library
from array import array
import math


class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return bytes(array(Vector2d.typecode, self))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def angle(self):
        return math.atan2(self.y, self.x)

# BEGIN VECTOR2D_V2_FORMAT
    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):  # <1>
            fmt_spec = fmt_spec[:-1]  # <2>
            coords = (abs(self), self.angle())  # <3>
            outer_fmt = '<{}, {}>'  # <4>
        else:
            coords = self  # <5>
            outer_fmt = '({}, {})'  # <6>
        components = (format(c, fmt_spec) for c in coords)  # <7>
        return outer_fmt.format(*components)  # <8>
# END VECTOR2D_V2_FORMAT

    @classmethod
    def frombytes(cls, octets):
        memv = memoryview(octets).cast(cls.typecode)
        return cls(*memv)

#Tests of ``format()`` with Cartesian coordinates:
v1 = Vector2d(3, 4)

print('format(v1) = {0}\n'.format(format(v1)))

print('format(v1, ".2f") = {0}\n'.format(format(v1, '.2f')))

print('format(v1, ".3e") = {0}\n'.format(format(v1, '.3e')))

#Tests of the ``angle`` method::

print('Vector2d(0, 0).angle() = {0}\n'.format(Vector2d(0, 0).angle()))

print('Vector2d(1, 0).angle() = {0}\n'.format(Vector2d(1, 0).angle()))

epsilon = 10**-8
print('abs(Vector2d(0, 1).angle() - math.pi/2) < epsilon = {0}\n'.format(abs(Vector2d(0, 1).angle() - math.pi/2) < epsilon))

print('abs(Vector2d(1, 1).angle() - math.pi/4) < epsilon = {0}\n'.format(abs(Vector2d(1, 1).angle() - math.pi/4) < epsilon))


#Tests of ``format()`` with polar coordinates:

print('format(Vector2d(1, 1), "p") = {0}\n'.format(format(Vector2d(1, 1), 'p')))  # doctest:+ELLIPSIS

print('format(Vector2d(1, 1), ".3ep") = {0}\n'.format(format(Vector2d(1, 1), '.3ep')))

print('format(Vector2d(1, 1), "0.5fp") = {0}\n'.format(format(Vector2d(1, 1), '0.5fp')))
