from util.console import console
from util.divider import print_divider
from statistic.statistic import Statistic
import numpy as np

module = 'RESULTS'


class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, n_epochs):
        print_divider('large')
        console(2, module, 'epochs: %d' % n_epochs)
        console(2, module, 'ave undiscounted return/epoch: %.3f +- %.3f' %
            (self.undiscounted_return.mean, self.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return/epoch: %.3f +- %.3f' %
            (self.discounted_return.mean, self.discounted_return.std_err()))
        console(2, module, 'ave time/epoch: %.3f' % self.time.mean)
        print_divider('medium')
