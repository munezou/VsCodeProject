# BEGIN TWILIGHT_BUS_CLASS
class TwilightBus_Modify:
    """A bus model that makes passengers vanish"""

    def __init__(self, passengers=None):
        if passengers is None:
            self.passengers = []  # <1>
        else:
            self.passengers = list(passengers)  #<2>

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)  # <3>
# END TWILIGHT_BUS_CLASS

basketball_team = ['Sue', 'Tina', 'Maya', 'Diana', 'Pat']
bus = TwilightBus_Modify(basketball_team)
bus.drop('Tina')
bus.drop('Pat')
print('basketball_team = {0}\n'.format(basketball_team))