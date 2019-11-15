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
        return (bytes([ord(self.typecode)]) +
                bytes(array(self.typecode, self)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

# BEGIN VECTOR2D_V1
    @classmethod  # <1>
    def frombytes(cls, octets):  # <2>
        typecode = chr(octets[0])  # <3>
        memv = memoryview(octets[1:]).cast(typecode)  # <4>
        return cls(*memv)  # <5>
# END VECTOR2D_V1

print('------------------------------------------------------------------------------------------------\n'
      '             9.3 Another version of the constructor　　　　　　　　　　　　　　　　　　　　　　　\n'
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

print('({0}, {1})\n'.format(bool(v1), bool(Vector2d(0, 0)))) # <9>

#Test of ``.frombytes()`` class method:

v1_clone = Vector2d.frombytes(bytes(v1))
print('v1_clone = {0}\n'.format(v1_clone))

print('v1 == v1_clone = {0}\n'.format(v1 == v1_clone))
