# common library
import weakref

# BEGIN CHEESE_CLASS
class Cheese:

    def __init__(self, kind):
        self.kind = kind

    def __repr__(self):
        return 'Cheese(%r)' % self.kind
# END CHEESE_CLASS

stock = weakref.WeakValueDictionary()
catalog = [Cheese('Red Leicester'), Cheese('Tilsit'), Cheese('Brie'), Cheese('Parmesan')]

for cheese in catalog:
    stock[cheese.kind] = cheese

print('sorted(stock.keys()) = \n{0}\n'.format(sorted(stock.keys())))

del catalog
print('sorted(stock.keys()) = {0}\n'.format(sorted(stock.keys())))

del cheese
print('sorted(stock.keys()) = {0}\n'.format(sorted(stock.keys())))