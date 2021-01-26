from util.step_record import StepRecord
import numpy as np


class History(list):
    """History object to store the step records list as a trajectory.

    This is a list in order of the step records of any trajectories. In FOGs, this
    represents the perfect game state in a game tree, and avoids calculation of
    observations every time the information state is obtained from the history.
    """

    def __init__(self, a=[], env=None):
        super().__init__(a)
        self._env = env
        self._children = {}

    def is_chance(self):
        """Whether is the chance node."""

        return self[-1].next_state.is_chance()

    def is_terminal(self):
        """Whether is the terminal node."""

        return self[-1].next_state.is_terminal()

    def current_player(self):
        """Get the current player of the history."""

        return self[-1].next_state.current_player()

    def legal_actions(self):
        """Return a list of actions are legal on this history."""

        return self[-1].next_state.legal_actions()

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        return self[-1].next_state.chance_outcomes()

    def get_return(self, discount=1):
        """Get discounted return of this trajectory."""

        get_return = 0
        factor = 1
        for step_record in self:
            get_return += factor * step_record.reward
            factor *= discount
        return get_return

    def child(self, action):
        """Get the child history given an action."""

        # Store the children as the values of a dict and the action as the keys
        if action.to_string() not in self._children.keys():
            step_record = self._env.step(self[-1].next_state, action)
            self._children[action.to_string()] = History(
                [record for record in self[:]], self._env)
            self._children[action.to_string()].append(step_record)

        return self._children[action.to_string()]

    def get_info_state(self):
        """Get the info states of two players corresponding to this history."""

        if not hasattr(self, '_info_state'):
            self._info_state = (InformationState(player=0, env=self._env),
                                InformationState(player=1, env=self._env))

            for player in [0, 1]:
                for record in self:
                    if record.action:  # not the first record in the history
                        # Append the action if player is the current player
                        if record.action.player == player:
                            self._info_state[player].append(record.action)
                        else:
                            self._info_state[player].append(None)
                    self._info_state[player].append(
                        (record.obs[player], record.obs[-1]))

        return self._info_state

    def get_public_state(self):
        """Get the public state corresponding to this history."""

        if not hasattr(self, '_public_state'):
            # List the public observation
            self._public_state = PublicState(
                [record.obs[-1] for record in self], env=self._env)

        return self._public_state

    def to_string(self):
        # Append the first world state string
        string = self[0].next_state.to_string()
        if len(self) > 1:
            string += ' -> '
            string += ' -> '.join(record.action.to_string() + ' -> ' +
                                  record.next_state.to_string() for record in self[1:])

        return string

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__


class InformationState(list):
    """Represents the node where players make decisions.

    This is a sequence of observations and actions of a single player 'i' like
    [O_i^0, a_i^0, O_i^1, a_i^1, ..., O_i^t] in FOGs.
    """

    def __init__(self, a=[], player=0, env=None):
        super().__init__(a)
        self.player = player
        self._env = env

    def get_all_histories(self):
        """Given a list of all histories, get a list of all possible histories
        corresponding to this information state."""

        if not hasattr(self, '_histories'):
            self._histories = [h for h in self._env.get_all_histories() if
                               h.get_info_state()[self.player] == self]

        return self._histories

    def get_public_state(self):
        """Get the public state corresponding to this information state."""

        if not hasattr(self, '_public_state'):
            self._public_state = PublicState(
                [item[-1] for item in self[::2]], env=self._env)

        return self._public_state

    @property
    def pot(self):
        return self[-1][-1].pot

    def to_string(self):
        def str_fun(item):
            if isinstance(item, tuple):
                return '; '.join(x.to_string() for x in item)
            else:
                return str(item)

        return ' -> '.join(map(str_fun, self))

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__


class PublicState(list):
    """Public state object defined in FOGs.

    This is a sequence of public observations like [O_pub^0, O_pub^1, ...,
    O_pub^t] in FOGs."""

    def __init__(self, a=[], env=None):
        super().__init__(a)
        self._env = env

    def get_all_histories(self):
        """Given a list of all histories, get a list of all possible histories
        corresponding to this public state."""

        if not hasattr(self, '_histories'):
            self._histories = [h for h in self._env.get_all_histories() if
                               h.get_public_state() == self]

        return self._histories

    # def get_all_infostates(self, infostate_list):
    #     """Given a list of all infostates, get a list of all possible infostates
    #     corresponding to this public state."""

    #     if not hasattr(self, '_infostates'):
    #         self._infostates = [
    #             s for s in infostate_list if s.get_public_state == self]
    #     return self._infostates

    def to_string(self):
        return ' -> '.join(o.to_string() for o in self)

    def __eq__(self, other):
        if isinstance(other, self.__class__) and len(self) == len(other):
            return all([x == y for x, y in zip(self, other)])
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__
