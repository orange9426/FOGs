from env.tiger.tiger_action import Action
from env.tiger.tiger_obs import Observation


class WorldState(object):
    """World state object of env: Kuhn Poker.

    A world state of Tiger is encoded as a single scalar that is
    0 when the tiger is behind the left door and 1 when the right,
    and is -1 when the game is over.
    """

    __name_dict = {0: 'left', 1: 'right', -1: 'terminal'}

    def __init__(self, encode):
        self.encode = encode
        self.name = WorldState.__name_dict[encode]

    def initial_obs(self):
        return Observation(-1)  # none

    def legal_actions(self):
        return [] if self.encode == -1 \
            else [Action(0), Action(1), Action(2)]

    def is_terminal(self):
        return self.encode == -1

    def to_string(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other
