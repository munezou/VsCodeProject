from functools import lru_cache
import urllib

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

[fib(n) for n in range(16)]

fib.cache_info()

data_today = datetime.date.today()

print   (
        '------------------------------------------------------------------------------------------------------\n'
    )

print(
    '       finished        functools.py                               ({0})                \n'.format(data_today)
    )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()