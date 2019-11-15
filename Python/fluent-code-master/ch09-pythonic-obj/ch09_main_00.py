# common library
from datetime import datetime
from array import array
import math


print('------------------------------------------------------------------------------------------------\n'
      '             9.4 classmethod and staticmethod 　　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')
class Demo:
    @classmethod
    def klassmeth(*args):
        return args
    
    @staticmethod
    def statmeth(*args):
        return args

print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth()))

print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth('spam')))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth()))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth('spam')))

print('------------------------------------------------------------------------------------------------\n'
      '             9.5 Output format                  　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')

# flot type specifier
brl = 1/2.43
print('brl = {0}\n'.format(brl))

print('format(brl, "0.4f") = {0}\n'.format(format(brl, '0.4f')))

print('1 BRL = {rate:0.2f} USD\n'.format(rate=brl))

# bit type specifier
print('format(42, "b") = {0}\n'.format(format(42, 'b')))

# % Format
print('format(2/3, ".1%") = {0}\n'.format(format(2/3, '.1%')))

# current date time
now = datetime.now()
print('now = {0}\n'.format(format(now, '%H:%M:%S')))

print('It is now {:%I:%M %p}\n'.format(now))

# BEGIN VECTOR2D_V0
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
    
    def __format__(self, fmt_spec=''):
        components = (format(c, fmt_spec) for c in self)
        return '({}, {})'.format(*components)
# END VECTOR2D_V0

v1 = Vector2d(3, 4)
print('format(v1) = \n{0}\n'.format(format(v1)))

print('format(v1, ".2f") = \n{0}\n'.format(format(format(v1, '.2f'))))

print('format(v1, ".3e") = \n{0}\n'.format(format(v1, '.3e')))

