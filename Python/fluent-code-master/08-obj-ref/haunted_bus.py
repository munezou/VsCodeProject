# BEGIN HAUNTED_BUS_CLASS
class HauntedBus:
    """A bus model haunted by ghost passengers"""

    def __init__(self, passengers=[]):  # <1>
        self.passengers = passengers  # <2>

    def pick(self, name):
        self.passengers.append(name)  # <3>

    def drop(self, name):
        self.passengers.remove(name)
# END HAUNTED_BUS_CLASS

# bigin to excute program.
bus1 = HauntedBus(['Alice', 'Bill'])
print('bus1.passengers = {0}\n'.format(bus1.passengers))

bus1.pick('Charlie')
bus1.drop('Alice')
print('bus1.passengers = {0}\n'.format(bus1.passengers))

bus2 = HauntedBus()
bus2.pick('Carrie')
print('bus2.passengers = {0}\n'.format(bus2.passengers))

bus3 = HauntedBus()
print('bus3.passengers = {0}\n'.format(bus3.passengers))

bus3.pick('Dave')
print('bus2.passengers = {0}\n'.format(bus2.passengers))

print('bus2.passengers is bus3.passengers = {0}\n'.format(bus2.passengers is bus3))

print('bus1.passengers = {0}\n'.format(bus1.passengers))

print('dir(HauntedBus.__init__) = \n{0}\n'.format(dir(HauntedBus.__init__)))

print('HauntedBus.__init__.__defaults__ = \n{0}\n'.format(HauntedBus.__init__.__defaults__))

print('HauntedBus.__init__.__defaults__[0] is bus2.passengers = {0}\n'.format(HauntedBus.__init__.__defaults__[0] is bus2.passengers))
# finish program