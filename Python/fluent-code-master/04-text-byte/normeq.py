"""
Utility functions for normalized Unicode string comparison.

Using Normal Form C, case sensitive:

    >>> s1 = 'café'
    >>> s2 = 'cafe\u0301'
    >>> s1 == s2
    False
    >>> nfc_equal(s1, s2)
    True
    >>> nfc_equal('A', 'a')
    False

Using Normal Form C with case folding:

    >>> s3 = 'Straße'
    >>> s4 = 'strasse'
    >>> s3 == s4
    False
    >>> nfc_equal(s3, s4)
    False
    >>> fold_equal(s3, s4)
    True
    >>> fold_equal(s1, s2)
    True
    >>> fold_equal('A', 'a')
    True

"""

from unicodedata import normalize

def nfc_equal(str1, str2):
    return normalize('NFC', str1) == normalize('NFC', str2)

def fold_equal(str1, str2):
    return (normalize('NFC', str1).casefold() ==
            normalize('NFC', str2).casefold())

print('----------------------------------------------------------------------\n'
      '  4.6.2  Utility functions for normalized Unicode string comparison.  \n'
      '----------------------------------------------------------------------\n')
      
s1 = 'café'
s2 = 'cafe\u0301'
print('s1 == s2, result = {0}'.format(s1 == s2))
print()
print('nfc_equal(s1, s2) = {0}'.format(nfc_equal(s1, s2)))
print()
print('nfc_equal("A", "a") = {0}'.format(nfc_equal('A', 'a')))
print()
s3 = 'Straße'
s4 = 'strasse'
print('s3 == s4, result = {0}'.format(s3 == s4))
print()
print('nfc_equal(s3, s4) = {0}'.format(nfc_equal(s3, s4)))
print()
print('fold_equal(s3, s4) = {0}'.format(fold_equal(s3, s4)))
print()
print('fold_equal(s1, s2) = {0}'.format(fold_equal(s1, s2)))
print()
print('fold_equal("A", "a") = {0}'.format(fold_equal('A', 'a')))
print()