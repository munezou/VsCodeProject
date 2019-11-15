'''
------------------------------------------------
 common lybrary
 -----------------------------------------------
 '''
#import os, sys
#sys.path.append(os.path.dirname(__file__))

#from registration import *
from collections import namedtuple

'''
------------------------------------------------------------------------------------------------------------------------
7.2 Decorator execution timing
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                    7.2 Decorator execution timing                                               \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
import registration

print('registration.registry = \n{0}'.format(registration.registry))
print()

'''
------------------------------------------------------------------------------------------------------------------------
7.3 Improving strategy patterns using decorators
------------------------------------------------------------------------------------------------------------------------
'''
print('-----------------------------------------------------------------------------------------------------------------\n'
      '                          7.3 Improving strategy patterns using decorators                                       \n'
      '-----------------------------------------------------------------------------------------------------------------\n')
promos = []  # <1>

def promotion(promo_func):  # <2>
    promos.append(promo_func)
    return promo_func

@promotion  # <3>
def fidelity(order):
    '''5% discount for customers with 1000 or more fidelity points'''
    return order.total() * .05 if order.customer.fidelity >= 1000 else 0

@promotion
def bulk_item(order):
    '''10% discount for each LineItem with 20 or more units'''
    discount = 0
    for item in order.cart:
        if item.quantity >= 20:
            discount += item.total() * .1
    return discount

@promotion
def large_order(order):
    '''7% discount for orders with 10 or more distinct items'''
    distinct_items = {item.product for item in order.cart}
    if len(distinct_items) >= 10:
        return order.total() * .07
    return 0

def best_promo(order):  # <4>
    '''Select best discount available'''
    return max(promo(order) for promo in promos)

# END STRATEGY_BEST4
