# common library
#import os, sys
#sys.path.append(os.path.dirname(__file__))

from collections import namedtuple
import inspect

#class
class promotions:
    @staticmethod
    def fidelity_promo(order):
        '''5% discount for customers with 1000 or more fidelity points'''
        return order.total() * .05 if order.customer.fidelity >= 1000 else 0

    @staticmethod
    def bulk_item_promo(order):
        '''10% discount for each LineItem with 20 or more units'''
        discount = 0
        for item in order.cart:
                if item.quantity >= 20:
                    discount += item.total() * .1
        return discount

    @staticmethod
    def large_order_promo(order):
        '''7% discount for orders with 10 or more distinct items'''
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
                return order.total() * .07
        return 0

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

promos = [func for name, func in inspect.getmembers(promotions, inspect.isfunction)]

def best_promo(order):
    '''Select best discount available'''
    return max(promo(order) for promo in promos)

'''
------------------------------------------------------------------------------------------------------------------------
6.1.4 Search for a strategy in a module.
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                          6.1.4-2 Search for a strategy in a module.                                             \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
joe = Customer('John Doe', 0)
ann = Customer('Ann Smith', 1100)
cart = [LineItem('banana', 4, .5),
        LineItem('apple', 10, 1.5),
        LineItem('watermellon', 5, 5.0)]

Order_joe_cart_fidelity = Order(joe, cart, promotions.fidelity_promo)
print('Order_joe_cart_fidelity = {0}'.format(Order_joe_cart_fidelity))

Order_ann_cart_fidelity = Order(ann, cart, promotions.fidelity_promo)
print('Order_ann_cart_fidelity = {0}'.format(Order_ann_cart_fidelity))

banana_cart = [LineItem('banana', 30, .5),
               LineItem('apple', 10, 1.5)]

Order_joe_banana_bulk = Order(joe, banana_cart, promotions.bulk_item_promo)
print('Order_joe_banana_bulk = {0}'.format(Order_joe_banana_bulk))

long_order = [LineItem(str(item_code), 1, 1.0) for item_code in range(10)]

Order_joe_long_large = Order(joe, long_order, promotions.large_order_promo)
print('Order_joe_long_large = {0}'.format(Order_joe_long_large))

Order_joe_cart_large = Order(joe, cart, promotions.large_order_promo)
print('Order_joe_cart_large = {0}'.format(Order_joe_cart_large))
print()

# BEGIN STRATEGY_BEST_TESTS

print('---< BEGIN STRATEGY_BEST_TESTS >---')

Order_joe_long_best = Order(joe, long_order, best_promo)
print('Order_joe_long_best = {0}'.format(Order_joe_long_best))

Order_joe_banana_best = Order(joe, banana_cart, best_promo)
print('Order_joe_banana_best = {0}'.format(Order_joe_banana_best))

Order_ann_cart_best = Order(ann, cart, best_promo)
print('Order_ann_cart_best = {0}'.format(Order_ann_cart_best))

# END STRATEGY_BEST_TESTS