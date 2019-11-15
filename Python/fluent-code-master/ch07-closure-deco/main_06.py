'''
--------------------------------------------
    common library
--------------------------------------------
'''
#import os, sys
#sys.path.append(os.path.dirname(__file__))

#from clockdeco import *

'''
------------------------------------------
    function
-----------------------------------------
'''
import time

def clock(func):
    def clocked(*args):
        t0 = time.time()
        result = func(*args)
        elapsed = time.time() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                             7.6 NONLOCAL declaration                                            \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

def make_averager():
    count = 0
    total = 0
    
    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count
    
    return averager

ave = make_averager()
print('avg(10) = {0}'.format(ave(10)))
print()

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                   7.7 Simple decorator implementation                                           \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
print('referance clockdeco_demo.py')