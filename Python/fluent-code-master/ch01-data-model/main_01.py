# common libraly
#import os, sys
#sys.path.append(os.path.dirname(__file__))

#from frenchdeck import Card, FrenchDeck
from random import choice
#from vector2d import Vector
import collections
from math import hypot

# class
Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]

class Vector:
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)


def main() -> None:
    beer_card = Card('7', 'diamonds')

    print('---------------< frenchdeck.py example >--------------------')
    print('beer_card : ', beer_card)

    print()

    deck = FrenchDeck();
    print ('length of deck = ', len(deck))

    print()

    print('-----------< choice card >----------------')
    for i in range(10):
        print(f'choice time[', i, '] =', choice(deck[i]))

    print()

    print('---< Display inner data in FrenchDeck >---')

    print('deck[0] = ', deck[0])
    print('deck[1] = ', deck[1])
    print('deck[2] = ', deck[2])
    print('deck[:3] = ', deck[:3])

    print()

    print('---< Element No .: 12 (index), 13 sheets at a time. >---')
    print('deck[12::13] = ', deck[12::13])

    print()

    print('--------< Output all card >-------')
    for card in deck[:]:
        print(card)

    print()

    print('---< Output reverse all card >---')
    for card in reversed(deck[:]):
        print(card)

    print()

    print ('-------< Output in order of card strength.  >-------')
    suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

    def spades_high(card):
        rank_value = FrenchDeck.ranks.index(card.rank)
        return rank_value * len(suit_values) + suit_values[card.suit]

    for card in sorted(deck[:], key=spades_high):
        print(card)

    print()

    print('---------------< vector.py example >--------------------')

    # additonal function
    v1 = Vector(2, 4)
    v2 = Vector(2, 1)

    print('v1 + v2 = ', v1 + v2)

    print()

    # abs function
    v = Vector(3, 4)
    print('abs(Vector(3, 4)) = ', abs(v))

    print()

    # Multiplication
    print('Vector(3, 4) * 3 = ', v * 3)

    print()

    # abs(Multiplication)
    print('abs(Vector(3, 4) * 3) = {0}'.format(abs(v * 3)))

if __name__ == '__main__':
    main()







