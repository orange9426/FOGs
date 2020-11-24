import env.environment as e


class WorldState(e.WorldState):
    """World state object of env: Tiger.

    A world state of Tiger is encoded as a single scalar that is 0 when the
    tiger is behind the left door and 1 when the right. It is -1 when it is
    the chance node and -2 when the game is over.
    """

    __name_dict = {0: 'left', 1: 'right', -1: 'chance', -2: 'terminal'}

    def __init__(self, encode):
        """Init the world state instance."""

        self.encode = encode
        self.player = 0 if encode == 1 else encode

    def legal_actions(self):
        """Return a list of actions are legal on this state."""

        if self.player == -2:  # is terminal
            return []
        elif self.player == -1:  # is chance
            return [Action(0, -1), Action(1, -1)]
        else:
            return [Action(0, 0), Action(1, 0), Action(2, 0)]

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        assert self.player == -1  # is chance
        action_list = self.legal_actions()
        prob_list = [1 / len(action_list) for _ in range(len(action_list))]
        return action_list, prob_list

    def to_string(self):
        """Return a string representing this world state."""

        return WorldState.__name_dict[self.encode]


class Action(e.Action):
    """Action object of env: Tiger.

    An action of Tiger is encoded as a single scalar that is 0, 1, 2 when
    act 'left', 'right', 'listen' respetively. In addition, the chance action
    dosn't include 'listen'.
    """

    __name_dict = {0: 'left', 1: 'right', 2: 'listen'}

    def __init__(self, encode, player=-1):
        """Init the action instance."""

        self.encode = encode
        self.player = player

    def to_string(self):
        """Return a string representing this action."""

        return Action.__name_dict[self.encode]


class Observation(e.Observation):
    """Observation object of env: Tiger.

    An observation of Tiger is encoded as a single scalar that is 0, 1 when
    tiger is listend at the left, right respetively, and is -1 when there is
    no observation.
    """

    __name_dict = {0: 'left', 1: 'right', -1: 'none'}

    def __init__(self, encode):
        """Init the observation instance."""

        self.encode = encode

    def to_string(self):
        """Return a string representing this private observation."""

        return Observation.__name_dict[self.encode]

    def possible_states(self):

        return [WorldState(0), WorldState(1)], [0.5, 0.5]
