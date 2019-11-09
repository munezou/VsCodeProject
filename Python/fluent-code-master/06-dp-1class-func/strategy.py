# strategy.py
# Strategy pattern -- function-based implementation

# BEGIN STRATEGY
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
            discount = self.promotion(self)  # <1>
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

# <2>

def fidelity_promo(order):  # <3>
    """5% discount for customers with 1000 or more fidelity points"""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0


def bulk_item_promo(order):
    """10% discount for each LineItem with 20 or more units"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount


def large_order_promo(order):
    """7% discount for orders with 10 or more distinct items"""
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0
# END STRATEGY


# BEGIN STRATEGY_TESTS
joe = Customer('John Doe', 0)  # <1>
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5), LineItem('apple', 10, 1.5), LineItem('watermellon', 5, 5.0)]

joe_cart_fidelity = Order(joe, cart, fidelity_promo)  # <2>
print('joe_cart_fidelity = \n{0}'.format(joe_cart_fidelity))
print()

ann_cart_fidelity = Order(ann, cart, fidelity_promo)
print('ann_cart_fidelity = \n{0}'.format(ann_cart_fidelity))
print()

banana_cart = [LineItem('banana', 30, .5), LineItem('apple', 10, 1.5)]

joe_banana_cart_bulk = Order(joe, banana_cart, bulk_item_promo)  # <3>
print('joe_banana_cart_bulk = \n{0}'.format(joe_banana_cart_bulk))
print()

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

joe_long_order_large_order = Order(joe, long_order, large_order_promo)
print('joe_long_order_large_order = \n{0}'.format(joe_long_order_large_order))
print()

joe_cart_large_order = Order(joe, cart, large_order_promo)
print('joe_cart_large_order = \n{0}'.format(joe_long_order_large_order))
print()
# END STRATEGY_TESTS
