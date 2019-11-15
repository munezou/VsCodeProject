# classic_strategy.py
# Strategy pattern -- classic implementation

# BEGIN CLASSIC_STRATEGY
from abc import ABC, abstractmethod
from collections import namedtuple

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
# END CLASSIC_STRATEGY

# BEGIN CLASSIC_STRATEGY_TESTS
joe = Customer('John Doe', 0)  # <1>
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5), LineItem('apple', 10, 1.5), LineItem('watermellon', 5, 5.0)]

joe_cart_Fidelity = Order(joe, cart, FidelityPromo())  # <3>
print('joe_cart_Fidelity = \n{0}'.format(joe_cart_Fidelity))
print()

ann_cart_Fidelity = Order(ann, cart, FidelityPromo())  # <4>
print('ann_cart_Fidelity = \n{0}'.format(ann_cart_Fidelity))
print()

banana_cart = [LineItem('banana', 30, .5), LineItem('apple', 10, 1.5)]

joe_banana_cart_BulkItem = Order(joe, banana_cart, BulkItemPromo())  # <6>
print('joe_banana_cart_BulkItem = \n{0}'.format(joe_banana_cart_BulkItem))
print()

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

joe_long_order_LargeOrder = Order(joe, long_order, LargeOrderPromo())  # <8>
print('joe_long_order_LargeOrder = \n{0}'.format(joe_long_order_LargeOrder))
print()

joe_cart_LargeOrder = Order(joe, cart, LargeOrderPromo())
print('joe_cart_LargeOrder = \n{0}'.format(joe_cart_LargeOrder))
print()
# END CLASSIC_STRATEGY_TESTS
