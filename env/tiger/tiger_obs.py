class Observation(object):
    """Observation object of env: Tiger.

    An observation of Tiger is encoded as a single scalar that is
    0, 1 when tiger is listend on the left, right respetively,
    and is -1 when there is no observation.
    """

    __name_dict = {0: 'left', 1: 'right', -1: 'none'}

    def __init__(self, encode):
        self.encode = encode
        self.name = Observation.__name_dict[encode]

    def to_string(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other
