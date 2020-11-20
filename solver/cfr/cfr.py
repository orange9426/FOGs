from statistic.record_history import History
from statistic.step_record import StepRecord

from solver.solver import Solver


class CFR(Solver):
    """
    Solver: CFR
    """

    def __init__(self, args):
        # Name of the solver
        self.name = args['solver']

    def reset_for_epoch(self):
        """Initial the solver before solving the game."""
        pass

    def solve_game(self, env):
        """Solve the entire game for one epoch."""
        pass
