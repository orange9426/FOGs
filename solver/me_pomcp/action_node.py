import numpy as np


class ActionNode(object):
    """A node that represents the action in the search tree."""

    def __init__(self, action, depth=-1):
        self.action = action
        self.depth = depth
        self.visit_count = 0
        self.total_reward = 0.0
        self.children = []

    def uct_value(self, parent_visit_count, uct_c):
        """Returns the UCT value of child."""
        if self.visit_count == 0:
            return float("inf")

        return self.total_reward / self.visit_count + uct_c * np.math.sqrt(
            np.math.log(parent_visit_count) / self.visit_count)

    def find_child(self, obs):
        """Returns the child obs node according to the given obs."""
        candi = [c for c in self.children if c.obs == obs]
        if candi:
            return np.random.choice(candi)
        else:
            return None

    @property
    def mean_value(self):
        return 0 if self.visit_count == 0 else self.total_reward / self.visit_count
