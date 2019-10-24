from functools import lru_cache
import urllib

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

[fib(n) for n in range(16)]

fib.cache_info()
