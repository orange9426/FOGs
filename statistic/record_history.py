from statistic.step_record import StepRecord
import numpy as np


class History(list):
    """History object to store the step records list as a trajectory.

    This is a list in order of the step records of any trajectories. In FOGs, this
    represents the perfect game state in a game tree, and avoids calculation of
    observations every time the information state is obtained from the history.
    """

    def __init__(self, a=[], env=None):
        super().__init__(a)
        self.env = env

    def child(self, action):
        """Get the child history given an action."""
        step_record = self.env.step(self[-1].next_state, action)
        history = History([record for record in self[:]], self.env)
        history.append(step_record)
        return history

    def get_info_state(self):
        """Get the info states of two players corresponding to the history."""

        info_state = (InformationState(player=0), InformationState(player=1))

        for player in [0, 1]:
            for record in self:
                if record.action:  # not the first record in the history
                    # Append the action if player is the current player
                    if record.action.player == player:
                        info_state[player].append(record.action)
                    else:
                        info_state[player].append(None)
                info_state[player].append((record.obs[player], record.obs[-1]))

        return info_state

    def get_public_state(self):
        """Get the public state corresponding to the history."""

        public_state = PublicState()

        # List the public observation
        for record in self:
            public_state.append(record.obs[-1])

        return public_state

    def undiscounted_return(self):
        """Get undiscounted return of the trajectory."""

        undiscounted_return = 0
        for step_record in self:
            undiscounted_return += step_record.reward
        return undiscounted_return

    def discounted_return(self, discount):
        """Get discounted return of the trajectory."""

        discounted_return = 0
        factor = 1
        for step_record in self:
            discounted_return += factor * step_record.reward
            factor *= discount
        return discounted_return

    def to_string(self):
        # Append the first world state string
        string = self[0].next_state.to_string() + ' -> '

        string += ' -> '.join(record.action.to_string() + ' -> ' +
                              record.state.to_string() for record in self[1:])
        return string

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other


class InformationState(list):
    """Represents the node where players make decisions.

    This is a sequence of observations and actions of a single player 'i' like
    [O_i^0, a_i^0, O_i^1, a_i^1, ..., O_i^t] in FOGs.
    """

    def __init__(self, a=[], player=0):
        super().__init__(a)
        self.player = player

    def get_public_state(self):
        """Get the public state corresponding to the information state."""
        public_state = PublicState()
        for i, item in enumerate(self):
            if i % 2 == 0:  # for observations
                public_state.append(item[-1])
        return public_state

    def to_string(self):
        def str_fun(item):
            if isinstance(item, tuple):
                return '; '.join(x.to_string() for x in item)
            elif item == None:
                return 'None'
            else:
                return item.to_string()

        return ' -> '.join(map(str_fun, self))

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other


class PublicState(list):
    """Public state object defined in FOGs."""

    def __init__(self, a=[]):
        super().__init__(a)

    def to_string():
        return ' -> '.join(o.to_string() for o in self)

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other
