#import os, sys
#sys.path.append(os.path.dirname(__file__))

#from clockdeco_demo import snooze
import functools

def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))

        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))

        arg_str = ', '.join(arg_lst)
        print('[%0.8fs] %s(%s) -> %s ' % (elapsed, name, arg_str, result))
        return result

    return clocked

@clock
def snooze(seconds):
    time.sleep(seconds)

#from generic import *

# BEGIN HTMLIZE
from functools import singledispatch
from collections import abc
import numbers
import html

@singledispatch  # <1>
def htmlize(obj):
    content = html.escape(repr(obj))
    return '<pre>{}</pre>'.format(content)

@htmlize.register(str)  # <2>
def _(text):            # <3>
    content = html.escape(text).replace('\n', '<br>\n')
    return '<p>{0}</p>'.format(content)

@htmlize.register(numbers.Integral)  # <4>
def _(n):
    return '<pre>{0} (0x{0:x})</pre>'.format(n)

@htmlize.register(tuple)  # <5>
@htmlize.register(abc.MutableSequence)
def _(seq):
    inner = '</li>\n<li>'.join(htmlize(item) for item in seq)
    return '<ul>\n<li>' + inner + '</li>\n</ul>'
# END HTMLIZE

print(htmlize({1, 2, 3}))
print()

print(htmlize((1, 2, 3)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
7.10 Parameterized decorator
------------------------------------------------------------------------------------------------------------------------
'''
#from registration import *
# BEGIN REGISTRATION

registry = []  # <1>

def register(func):  # <2>
    print('running register(%s)' % func)  # <3>
    registry.append(func)  # <4>
    return func  # <5>

@register  # <6>
def f1():
    print('running f1()')

@register
def f2():
    print('running f2()')

def f3():  # <7>
    print('running f3()')

def main():  # <8>
    print('running main()')
    print('registry ->', registry)
    f1()
    f2()
    f3()
# END REGISTRATION

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                            7.10 Parameterized decorator                                                         \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
main()
print()

'''
------------------------------------------------------------------------------------------------------------------------
7.10.1 Registered decorator parameterization
------------------------------------------------------------------------------------------------------------------------
'''
#from registration_param import *
# BEGIN REGISTRATION_PARAM

registry = set()  # <1>

def register(active=True):  # <2>
    def decorate(func):  # <3>
        print('running register(active=%s)->decorate(%s)'
              % (active, func))
        if active:   # <4>
            registry.add(func)
        else:
            registry.discard(func)  # <5>

        return func  # <6>
    return decorate  # <7>

@register(active=True)  # <8>
def f1():
    print('running f1()')

@register(active=False)  # <9>
def f2():
    print('running f2()')

@register(active=True)
def f3():
    print('running f3()')

# END REGISTRATION_PARAM

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                        7.10.1 Registered decorator parameterization    �@                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('f1() = {0}'.format(f1))
print('registtry = {0}'.format(registry))
print()

print('f2() = {0}'.format(f2))
print('registtry = {0}'.format(registry))
print()

print('f3() = {0}'.format(f3))
print('registtry = {0}'.format(registry))
print()

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                             7.10.2 clock decorator parameterization    �@                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
#from clockdeco_param import clock
import time

# BEGIN CLOCKDECO_PARAM
DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result}'

def clock(fmt=DEFAULT_FMT):  # <1>
    def decorate(func):      # <2>
        def clocked(*_args): # <3>
            t0 = time.time()
            _result = func(*_args)  # <4>
            elapsed = time.time() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in _args)  # <5>
            result = repr(_result)  # <6>
            print(fmt.format(**locals()))  # <7>
            return _result  # <8>
        return clocked  # <9>
    return decorate  # <10>
# END CLOCKDECO_PARAM

clock()
clock('{name}: {elapsed}')(time.sleep)(.2)  # doctest: +ELLIPSIS
clock('{name}({args}) dt={elapsed:0.3f}s')(time.sleep)(.2)
