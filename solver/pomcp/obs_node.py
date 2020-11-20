import numpy as np


class ObservationNode(object):
    """A node that represents the observation in the search tree."""

    def __init__(self, obs, depth=-1):
        self.obs = obs
        self.depth = depth
        self.visit_count = 0
        self.particle_bin = []
        self.children = []

    def find_child(self, action):
        """Returns the child action node according to the given action."""
        candi = [c for c in self.children if c.action == action]
        if candi:
            return np.random.choice(candi)
        else:
            return None

    def best_child(self):
        """Returns the best child in order of the sort key."""
        return max(self.children, key=ObservationNode.sort_key)

    def sort_key(self):
        """The key function for searching best child."""
        return (self.visit_count, self.total_reward)
