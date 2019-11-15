# common library
import copy

# BEGIN BUS_CLASS
class Bus:

    def __init__(self, passengers=None):
        if passengers is None:
            self.passengers = []
        else:
            self.passengers = list(passengers)

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)
# END BUS_CLASS


bus1 = Bus(['Alice', 'Bill', 'Claire', 'David'])
bus2 = copy.copy(bus1)
bus3 = copy.deepcopy(bus1)

print('bus1 = \n{0}\n'.format(bus1))
print('bus2 = \n{0}\n'.format(bus2))
print('bus3 = \n{0}\n'.format(bus3))

bus1_drop_Bill = bus1.drop('Bill')
bus2_passengers = bus2.passengers
print('bus2_passenger = \n{0}\n'.format(bus2_passengers))
print('bus3.passengers = \n{0}\n'.format(bus3.passengers))
