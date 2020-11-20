from env.kuhn_poker.kuhn_poker_action import Action
from env.kuhn_poker.kuhn_poker_obs import PrivateObservation
from env.kuhn_poker.kuhn_poker_obs import PublicObservation


class WorldState(object):
    """World state object of env: Kuhn Poker.

    A world state of Kuhn Poker is encoded as [[h1, h2], [b1, b2], turn],
    where h1, h2 represent the hands of two players respectively, and
    b1, b2 represent the total bet of two players. 'turn' indicates which
    player is currently playing including the chance as -1, and when turn 
    == -2, it means that the game is over.
    """

    __hand_dict = {0: 'J', 1: 'Q', 2: 'K', -1: '?'}

    def __init__(self, encode):
        self.encode = encode
        self.player = encode[-1]
        self.name = '[%s, %s], [%d, %d], %d' % \
            (WorldState.__hand_dict[encode[0][0]],
             WorldState.__hand_dict[encode[0][1]],
             encode[1][0], encode[1][1], encode[2])

    def is_chance(self):
        return self.player == -1

    def is_terminal(self):
        return self.player == -2

    def current_player(self):
        return self.player

    def legal_actions(self):
        if self.player == -1:  # is chance
            return [Action([i, (i + j) % 3])
                    for i in range(3) for j in range(1, 3)]
        elif self.player == -2:  # is terminal
            return []
        else:
            return [Action(0, self.player), Action(1, self.player)]

    def chance_outcome(self):
        assert self.player == -1  # is chance
        action_list = self.legal_actions()
        prob_list = [1 / len(action_list) for _ in len(action_list)]
        return action_list, prob_list

    def to_string(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other
