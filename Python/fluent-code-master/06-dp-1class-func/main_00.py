'''
------------------------------------------------------------------------------------------------------------------------
6.1 Strategy pattern as a refactoring case study
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                             6.1.1 Typical Strategy pattern                                                      \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
# common library
#import os, sys
#sys.path.append(os.path.dirname(__file__))

#from classic_strategy import *
from abc import ABC, abstractmethod
from collections import namedtuple

# class
Customer = namedtuple('Customer', 'name fidelity')


class LineItem:

    def __init__(self, product, quantity, price):
        self.product = product
        self.quantity = quantity
        self.price = price

    def total(self):
        return self.price * self.quantity


class Order:  # the Context

    def __init__(self, customer, cart, promotion=None):
        self.customer = customer
        self.cart = list(cart)
        self.promotion = promotion

    def total(self):
        if not hasattr(self, '__total'):
            self.__total = sum(item.total() for item in self.cart)
        return self.__total

    def due(self):
        if self.promotion is None:
            discount = 0
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())


class Promotion(ABC):  # the Strategy: an Abstract Base Class

    @abstractmethod
    def discount(self, order):
        """Return discount as a positive dollar amount"""


class FidelityPromo(Promotion):  # first Concrete Strategy
    """5% discount for customers with 1000 or more fidelity points"""

    def discount(self, order):
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0


class BulkItemPromo(Promotion):  # second Concrete Strategy
    """10% discount for each LineItem with 20 or more units"""

    def discount(self, order):
        discount = 0
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * .1
        return discount


class LargeOrderPromo(Promotion):  # third Concrete Strategy
    """7% discount for orders with 10 or more distinct items"""

    def discount(self, order):
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * .07
        return 0

joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)

cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe = Order(joe, cart, FidelityPromo())
print('Order_joe = \n {0}'.format(Order_joe))
print()

Order_ann = Order(ann, cart, FidelityPromo())
print('Order_ann = \n {0}'.format(Order_ann))
print()

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order_joe_2 = Order(joe, banana_cart, BulkItemPromo())
print('Order_joe_2 = \n{0}'.format(Order_joe_2))
print()

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]
Order_joe_3 = Order(joe, long_order, LargeOrderPromo())
print('Order_joe_3 = \n{0}'.format(Order_joe_3))
print()

Order_joe_4 = Order(joe, cart, LargeOrderPromo())
print('Order_joe_4 = \n{0}'.format(Order_joe_4))
print()
