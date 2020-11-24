import env.environment as e


class WorldState(e.WorldState):
    """World state object of env: Kuhn Poker.

    A world state of Kuhn Poker is encoded as [[h1, h2], [b1, b2], turn],
    where h1, h2 represent the hands of two players respectively, and
    b1, b2 represent the total bet of two players. 'turn' indicates which
    player is currently playing including the chance as -1, and when turn 
    == -2, it means that the game is over.
    """

    __hand_dict = {0: 'J', 1: 'Q', 2: 'K', -1: '?'}

    def __init__(self, encode):
        """Init the world state instance."""

        self.encode = encode
        self.player = encode[-1]

    def legal_actions(self):
        """Return a list of actions are legal on this state."""

        if self.player == -2:  # is terminal
            return []
        elif self.player == -1:  # is chance
            return [Action([i, (i + j) % 3])
                    for i in range(3) for j in range(1, 3)]
        else:
            return [Action(0, self.player), Action(1, self.player)]

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        assert self.player == -1  # is chance
        action_list = self.legal_actions()
        prob_list = [1 / len(action_list) for _ in range(len(action_list))]
        return action_list, prob_list

    def to_string(self):
        """Return a string representing this world state."""

        return '[%s, %s], [%d, %d], %d' % \
            (WorldState.__hand_dict[self.encode[0][0]],
             WorldState.__hand_dict[self.encode[0][1]],
             self.encode[1][0], self.encode[1][1], self.encode[2])


class Action(e.Action):
    """Action object of env: Kuhn Poker.

    A player's action of Kuhn Poker is encoded as a single scalar that
    is 0, 1 when act pass, bet respetively. A chance's action is encoded
    as [h1, h2] that indicates the deal.
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
            return ', '.join(Action.__hand_dict[x] for x in self.encode)


class PrivateObservation(e.Observation):
    """Private observation object of env: Kuhn Poker.

    A private observation of Kuhn Poker is encoded as a single scalar
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
    """Public observation object of env: Kuhn Poker.

    A public observations of Kuhn Poker is encoded as [b1, b2], where
    b1, b2 represent the total bet of two players at the moment.
    """

    def __init__(self, encode):
        """Init the public observation instance."""

        self.encode = encode

    def to_string(self):
        """Return a string representing this public observation."""

        return ', '.join(str(x) for x in self.encode)
