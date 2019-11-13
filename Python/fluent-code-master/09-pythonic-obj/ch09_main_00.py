# common library




print('------------------------------------------------------------------------------------------------\n'
      '             9.4 classmethod and staticmethod 　　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')
class Demo:
    @classmethod
    def klassmeth(*args):
        return args
    
    @staticmethod
    def statmeth(*args):
        return args

print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth()))

print('Demo.klassmeth() = \n{0}\n'.format(Demo.klassmeth('spam')))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth()))

print('Demo.statmeth() = \n{0}\n'.format(Demo.statmeth('spam')))

print('------------------------------------------------------------------------------------------------\n'
      '             9.5 Output format                  　　　　　　　　　　　　　　　　　　　　　　　　\n'
      '------------------------------------------------------------------------------------------------\n')

