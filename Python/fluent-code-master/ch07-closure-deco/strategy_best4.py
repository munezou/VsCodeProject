# strategy_best4.py
# Strategy pattern -- function-based implementation
# selecting best promotion from list of functions
# registered by a decorator

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
            discount = self.promotion(self)
        return self.total() - discount

    def __repr__(self):
        fmt = '<Order total: {:.2f} due: {:.2f}>'
        return fmt.format(self.total(), self.due())

# BEGIN STRATEGY_BEST4

promos = []  # <1>

def promotion(promo_func):  # <2>
    promos.append(promo_func)
    return promo_func

@promotion  # <3>
def fidelity(order):
    """5% discount for customers with 1000 or more fidelity points"""
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

@promotion
def bulk_item(order):
    """10% discount for each LineItem with 20 or more units"""
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount

@promotion
def large_order(order):
    """7% discount for orders with 10 or more distinct items"""
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0

def best_promo(order):  # <4>
    """Select best discount available
    """
    return max(promo(order) for promo in promos)

# END STRATEGY_BEST4

# BEGIN STRATEGY_BEST_TESTS
joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe_cart_fidelity = Order(joe, cart, fidelity)
print('Order_joe_cart_fidelity = {0}'.format(Order_joe_cart_fidelity))

Order_ann_cart_fidelity = Order(ann, cart, fidelity)
print('Order_ann_cart_fidelity = {0}'.format(Order_ann_cart_fidelity))

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

joe_banana_cart_bulk = Order(joe, banana_cart, bulk_item)
print('joe_banana_cart_bulk = {0}'.format(joe_banana_cart_bulk))
print()

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

Order_joe_long_best = Order(joe, long_order, best_promo)
print('Order_joe_long_best = {0}'.format(Order_joe_long_best))

Order_joe_banana_best = Order(joe, banana_cart, best_promo)
print('Order_joe_banana_best = {0}'.format(Order_joe_banana_best))

Order_ann_cart_best = Order(ann, cart, best_promo)
print('Order_ann_cart_best = {0}'.format(Order_ann_cart_best))
print()             
# END STRATEGY_BEST_TESTS