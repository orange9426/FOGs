import env.environment as e
import numpy as np


class WorldState(e.WorldState):
    """World state object of env: Leduc Poker.

    A world state of Leduc Poker is encoded as [[h1, h2, hp], [b1, b2], turn],
    where h1, h2, hp represent the hands of two players and public respectively,
    and b1, b2 represent the total bet of two players. 'turn' indicates which
    player is currently playing including the chance as -1, and when turn 
    == -2, it means that the game is over.
    """

    __hand_dict = {0: 'J', 1: 'Q', 2: 'K', -1: '?'}

    def __init__(self, encode):
        """Init the world state instance."""

        self.encode = encode
        self.player = encode[-1]

        self.hand = self.encode[0][:2]
        self.bet = self.encode[1]
        self.pub = self.encode[0][-1]
        self.phase = 1 if self.pub != -1 or \
            (self.player == -1 and self.hand != [-1, -1]) else 0

    def legal_actions(self):
        """Return a list of actions that are legal on this state."""

        if self.player == -2:  # is terminal
            return []
        elif self.player == -1:  # is chance
            if self.phase == 0:  # private hands chance
                return [Action([i, j]) for i in range(3) for j in range(3)]
            else:  # public hands chance
                return [Action(i) for i in range(3)]
        else:
            return [Action(0, self.player), Action(1, self.player)]

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        assert self.is_chance()

        action_list = self.legal_actions()
        if self.phase == 0:  # private hands chance
            prob_list = [1 if a.encode[0] == a.encode[1] else 2
                         for a in action_list]
        else:
            prob_list = [2 - int(self.hand[0] == a.encode) -
                         int(self.hand[1] == a.encode) for a in action_list]
        prob_list = np.array(prob_list)
        prob_list = prob_list / np.sum(prob_list)

        return action_list, prob_list

    @property
    def winner(self):
        """Return the winner for a terminal world state."""

        assert self.is_terminal()

        if self.bet[0] != self.bet[1]:  # the player who passed will lose
            return 0 if self.bet[0] > self.bet[1] else 1
        else:  # campare the hands
            if self.hand[0] == self.hand[1]:
                return -1  # draw
            elif self.hand[0] == self.pub:
                return 0
            elif self.hand[1] == self.pub:
                return 1
            else:
                return 0 if self.hand[0] > self.hand[1] else 1

    def to_string(self):
        """Return a string representing this world state."""

        return '[%s, %s, %s], [%d, %d], %d' % \
            (WorldState.__hand_dict[self.hand[0]],
             WorldState.__hand_dict[self.hand[1]],
             WorldState.__hand_dict[self.pub],
             self.bet[0], self.bet[1], self.player)


class Action(e.Action):
    """Action object of env: Leduc Poker.

    A player's action of Leduc Poker is encoded as a single scalar that
    is 0, 1 when act pass, bet respetively. A chance's action is encoded
    as [h1, h2] or hp that indicates the deal.
    """

    __act_dict = {0: 'pass', 1: 'bet'}
    __hand_dict = {0: 'J', 1: 'Q', 2: 'K'}

    def __init__(self, encode, player=-1):
        """Init the action instance."""

        self.encode = encode
        self.player = player

    def to_string(self):
        """Return a string representing this action."""

        if self.player != -1:  # is not chance
            return Action.__act_dict[self.encode]
        else:  # is chance
            if isinstance(self.encode, list):
                return ', '.join(Action.__hand_dict[x] for x in self.encode)
            else:
                return Action.__hand_dict[self.encode]


class PrivateObservation(e.Observation):
    """Private observation object of env: Leduc Poker.

    A private observation of Leduc Poker is encoded as a single scalar
    that indicates the hand of a single player, and when it is -1, it
    means that the hand is unknown.
    """

    __hand_dict = {0: 'J', 1: 'Q', 2: 'K', -1: '?'}

    def __init__(self, encode, player):
        """Init the private observation instance."""

        self.encode = encode
        self.player = player

    def to_string(self):
        """Return a string representing this private observation."""

        return PrivateObservation.__hand_dict[self.encode]


class PublicObservation(e.Observation):
    """Public observation object of env: Leduc Poker.

    A public observations of Leduc Poker is encoded as [[b1, b2], hp], where
    b1, b2 represent the total bet of two players at the moment and hp is the
    public hand which is -1 when it is unknown.
    """

    __hand_dict = {0: 'J', 1: 'Q', 2: 'K', -1: '?'}

    def __init__(self, encode):
        """Init the public observation instance."""

        self.encode = encode
        self.bet = encode[0]
        self.pub = encode[1]

    def to_string(self):
        """Return a string representing this public observation."""

        return '[%s], %s' % (', '.join(str(x)for x in self.bet),
                             PublicObservation.__hand_dict[self.pub])
