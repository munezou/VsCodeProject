'''
---------------------------------------------------------
    common library
---------------------------------------------------------
'''
import weakref

print('-----------------------------------------------------------------\n'
      '                 8.6.2 Weak reference constraints　              \n'
      '-----------------------------------------------------------------\n')
class MyList(list):
    ''' list subclass whose instances may be weakly reference. '''

a_list = MyList(range(10))

# a_list can be target of a weak reference
wref_to_a_list = weakref.ref(a_list)

print('wref_to_a_list = \n{0}\n'.format(wref_to_a_list))

print('-----------------------------------------------------------------\n'
      '                 8.7 Magic used by Python invariant              \n'
      '-----------------------------------------------------------------\n')

t1 = (1, 2, 3)
t2 = tuple(t1)
print('t2 is t1 = {0}\n'.format(t2 is t1))

t3 = t1[:]
print('t3 is t1 = {0}\n'.format(t3 is t1))

# String literals can also create shared objects.
t1 = (1, 2, 3)
t3 = (1, 2, 3)
print('t3 is t1 = {0}\n'.format(t3 is t1))

s1 = 'ABC'
s2 = 'ABC'
print('s2 is s1 = {0}\n'.format(s2 is s1))