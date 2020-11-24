from policy.step_record import StepRecord
from policy.history import History
import abc


class Environment(abc.ABC):
    """Abstract class of all kinds of environments."""

    @abc.abstractmethod
    def initial_state(self):
        """Get a new initial world state."""
        pass

    @abc.abstractmethod
    def initial_obs(self):
        """Get new initial observations."""
        pass

    @abc.abstractmethod
    def step(self):
        """Get the step result given a world state and an action."""
        pass

    def initial_history(self):
        """Get new initial history with the initial record."""

        # Append the init state and the init obs to the init history
        return History([StepRecord(next_state=self.initial_state(),
                                   obs=self.initial_obs())], self)

    def get_all_histories(self):
        """Return a list of all possible histories in the game."""

        # Breadth-first to traverse the game tree
        history_list = []
        history_queue = [self.initial_history()]
        while history_queue:
            history = history_queue.pop(0)
            history_list.append(history)
            if not history[-1].is_terminal:
                history_queue += [history.child(action) for action in
                                  history[-1].next_state.legal_actions()]

        return history_list

    def __str__(self):
        return self.name

    __repr__ = __str__


class WorldState(abc.ABC):
    """Abstract class of all kinds of world states."""

    @abc.abstractmethod
    def legal_actions(self):
        """Return a list of actions are legal on this state."""
        pass

    @abc.abstractmethod
    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""
        pass

    @abc.abstractmethod
    def to_string(self):
        """Return a string representing this world state."""

    def is_chance(self):
        """Whether is the chance node."""
        return self.player == -1

    def is_terminal(self):
        """Whether is the terminal node."""
        return self.player == -2

    def current_player(self):
        """Return the current player."""
        return self.player

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__


class Action(abc.ABC):
    """Abstract class of all kinds of actions."""

    @abc.abstractmethod
    def to_string(self):
        """Return a string representing this action."""
        pass

    def is_chance(self):
        """Whether is the chance action."""
        return self.player == -1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__


class Observation(abc.ABC):
    """Abstract class of all kinds of observations."""

    @abc.abstractmethod
    def to_string(self):
        """Return a string representing this observation."""
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.encode == other.encode
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.to_string()

    __repr__ = __str__
