'''
------------------------------------------------------------------------------------------
Revised clockdeco
------------------------------------------------------------------------------------------
'''
#import os, sys
#sys.path.append(os.path.dirname(__file__))

import time
import functools

#from clockdeco_modify import clock
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

@clock
def snooze(second):
    time.sleep(second)

@clock
def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)

if __name__ == '__main__':
    print('*' * 40, 'Calling snooze(.123)')
    snooze(.123)
    print('*' * 40, 'Calling snooze(6)')
    print('6! =', factorial(6))

'''
------------------------------------------------------------------------------------------------------------------------
7.8 Standard library decorator
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                           7.8.1 Memoization using functools.lru_cache                                           \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
#from fibo_demo import *
@clock
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-2) + fibonacci(n-1)

print('---< no using lru_cache >---')
fibonacci(6)
print()

#from fibo_demo_lru import *
@functools.lru_cache() # <1>
@clock  # <2>
def fibonacci_lru(n):
    if n < 2:
        return n
    return fibonacci_lru(n-2) + fibonacci_lru(n-1)

print('---< using lru_cache >---')
fibonacci_lru(30)

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                         7.8.2 Single dispatch generic function                                                  \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
import html

def htmlize(obj):
    contents = html.escape(repr(obj))
    return '<pre>{0}</pre>'.format(contents)

print(htmlize({1, 2, 3}))
print()

print(htmlize(abs))
print()

print(htmlize('Heimich & Co.\n- a game'))
print()

print(htmlize(42))
print()

print(htmlize(['alpha', 66, {3, 2, 1}]))
print()