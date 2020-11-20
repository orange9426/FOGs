class Action(object):
    """Action object of env: Kuhn Poker.

    A player's action of Kuhn Poker is encoded as a single scalar that
    is 0, 1 when act pass, bet respetively. A chance's action is encoded
    as [h1, h2] that indicates the deal.
    """

    __act_dict = {0: 'pass', 1: 'bet'}
    __hand_dict = {0: 'J', 1: 'Q', 2: 'K'}

    def __init__(self, encode, player=-1):
        self.encode = encode
        self.player = player
        if player != -1:  # is not chance
            self.name = Action.__act_dict[encode]
        else:  # is chance
            self.name = ', '.join(Action.__hand_dict[x] for x in encode)

    def is_chance(self):
        return self.player == -1

    def to_string(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other
