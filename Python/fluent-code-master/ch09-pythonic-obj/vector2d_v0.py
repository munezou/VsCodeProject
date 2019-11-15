"""
---------------------------------
A 2-dimensional vector class
---------------------------------
"""

# BEGIN VECTOR2D_V0
from array import array
import math


class Vector2d:
    typecode = 'd'  # <1>

    def __init__(self, x, y):
        self.x = float(x)    # <2>
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))  # <3>

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)  # <4>

    def __str__(self):
        return str(tuple(self))  # <5>

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +  # <6>
                bytes(array(self.typecode, self)))  # <7>

    def __eq__(self, other):
        return tuple(self) == tuple(other)  # <8>

    def __abs__(self):
        return math.hypot(self.x, self.y)  # <9>

    def __bool__(self):
        return bool(abs(self))  # <10>
# END VECTOR2D_V0

# BEGIN VECTOR2D_V0_DEMO
print('------------------------------------------------------------------------------------------------\n'
      '             9.2 Revisiting the vector class　　　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')

v1 = Vector2d(3, 4)
print('v1.x = {0}\nV1.y = {1}\n'.format(v1.x, v1.y))  # <1>

x, y = v1  # <2>
print('({0}, {1})'.format(x, y))

print('v1 = {0}\n'.format(v1))  # <3>

v1_clone = eval(repr(v1))  # <4>
print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))
print(v1)  # <6>
print()

octets = bytes(v1)  # <7>
print('octets = {0}\n'.format(octets))

print('abs(v1) = {0}\n'.format(abs(v1))) # <8>

print('({0}, {1})'.format(bool(v1), bool(Vector2d(0, 0)))) # <9>

# END VECTOR2D_V0_DEMO
