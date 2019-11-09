"""
# BEGIN BINGO_DEMO

>>> bingo = BingoCage(range(3))
>>> bingo.pick()
1
>>> bingo()
0
>>> callable(bingo)
True

# END BINGO_DEMO

"""

# BEGIN BINGO

import random

class BingoCage:

    def __init__(self, items):
        self._items = list(items)  # <1>
        random.shuffle(self._items)  # <2>

    def pick(self):  # <3>
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')  # <4>

    def __call__(self):  # <5>
        return self.pick()

# END BINGO

# BEGIN BINGO_DEMO

bingo = BingoCage(range(3))
bingo_pick = bingo.pick()
print('bingo_pick = {0}'.format(bingo_pick))
print()

bingo_callable = callable(bingo)
print('bingo_callable = {0}'.format(bingo_callable))
print()
# END BINGO_DEMO
