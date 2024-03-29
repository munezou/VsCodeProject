import os, sys
sys.path.append(os.path.dirname(__file__))

import functools
from clockdeco import clock

@functools.lru_cache() # <1>
@clock  # <2>
def fibonacci_lru(n):
    if n < 2:
        return n
    return fibonacci_lru(n-2) + fibonacci_lru(n-1)

if __name__=='__main__':
    print(fibonacci_lru(6))
