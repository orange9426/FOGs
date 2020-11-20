import abc


class Solver(object):
    """Abstract class of all kinds of solvers."""

    @abc.abstractmethod
    def reset_for_epoch(self):
        pass

    @abc.abstractmethod
    def solve_game(self):
        pass
