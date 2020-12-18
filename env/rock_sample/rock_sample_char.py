import env.environment as e


class WorldState(e.WorldState):
    """World state object of env: Rock Sample.

    A world state of Rock Sample is a map encoded as a 2-dim numpy array, that 'o'
    indicates the goot rock, 'x' indicates the bad rock, 'p' indicates the player
    which can follow 'o' or 'x', and '.' indicates the empty grid.
    """

    def __init__(self, encode):
        """Init the world state instance."""

        self.encode = encode
        self.map = encode
        self.pos = None
        self.rock_list = []

        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j][-1] == 'p':
                    self.pos = (i, j)
                if self.map[i][j][0] == 'o':
                    self.rock_list.append(((i, j), 'good'))
                elif self.map[i][j][0] == 'x':
                    self.rock_list.append(((i, j), 'bad'))
                elif self.map[i][j][0] == '-':
                    self.rock_list.append(((i, j), 'picked'))

        if self.pos is None:
            self.player = -1  # chance
        elif self.pos[1] == len(self.map[0]) - 1:
            self.player = -2  # terminal
        else:
            self.player = 0

    def legal_actions(self):
        """Return a list of actions are legal on this state."""

        if self.player == -2:  # is terminal
            return []
        elif self.player == -1:  # is chance
            # 2^n_rock actions for all possibilities of rocks for good or bad
            return [Action(a, -1) for a in range(2 ** len(self.rock_list))]
        else:
            action_list = []
            if self.pos[1] < len(self.map[0]) - 1:
                action_list.append(Action(0, 0))  # east
            if self.pos[0] < len(self.map) - 1:
                action_list.append(Action(1, 0))  # south
            if self.pos[1] > 0:
                action_list.append(Action(2, 0))  # west
            if self.pos[0] > 0:
                action_list.append(Action(3, 0))  # north
            if self.map[self.pos[0]][self.pos[1]] is 'op' or self.map[self.pos[0]][self.pos[1]] is 'xp':
                action_list.append(Action(4, 0))  # sample
            action_list += [Action(a, 0) for a in range(5, 5 + len(self.rock_list))]

            return action_list

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        assert self.player == -1  # is chance
        action_list = self.legal_actions()
        prob_list = [1 / len(action_list) for _ in range(len(action_list))]
        return action_list, prob_list

    def to_string(self):
        """Return a string representing this world state."""

        return '\n' + '\n'.join('\t'.join(self.map[i]) for i in range(len(self.map)))


class Action(e.Action):
    """Action object of env: Rock Sample.

    An action of Rock Sample is encoded as a single scalar that is 0, 1, 2, 3, 4, 5,
    ..., n_rock+5 when act 'east', 'south', 'west', 'north', 'sample', 'check rock 0',
    ..., check rock n_rock respetively. In addition, there are 2^n_rock actions for
    beginning chance node.
    """

    __name_dict = {0: 'east', 1: 'south', 2: 'west', 3: 'north', 4: 'sample'}

    def __init__(self, encode, player=-1):
        """Init the action instance."""

        self.encode = encode
        self.player = player

    def to_string(self):
        """Return a string representing this action."""

        if self.player == -1:  # chance
            return str(self.encode)
        else:
            if self.encode <= 4:
                return Action.__name_dict[self.encode]
            else:
                return 'check rock No.%d' % (self.encode - 5)


class Observation(e.Observation):
    """Observation object of env: Rock Sample.

    An observation of Rock Sample is encoded as a single scalar that is 0, 1,
    when the rock is checked as bad, good respetively, and is -1 when there is
    no observation.
    """

    __name_dict = {0: 'bad', 1: 'good', -1: 'none'}

    def __init__(self, encode):
        """Init the observation instance."""

        self.encode = encode

    def to_string(self):
        """Return a string representing this private observation."""

        return Observation.__name_dict[self.encode]
