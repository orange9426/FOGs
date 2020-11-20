import abc


class Environment(object):
    """Abstract class of all kinds of environments."""

    @abc.abstractmethod
    def initial_states(self):
        pass

    @abc.abstractmethod
    def new_initial_state(self):
        pass

    @abc.abstractmethod
    def get_all_states(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass
