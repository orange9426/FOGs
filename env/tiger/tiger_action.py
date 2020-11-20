class Action(object):
    """Action object of env: Tiger.

    An action of Tiger is encoded as a single scalar that is
    0, 1, 2 when act left, right, listen respetively.
    """

    __name_dict = {0: 'left', 1: 'right', 2: 'listen'}

    def __init__(self, encode):
        self.encode = encode
        self.name = Action.__name_dict[encode]

    def to_string(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other
