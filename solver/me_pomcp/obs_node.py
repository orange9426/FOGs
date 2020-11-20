import numpy as np


class ObservationNode(object):
    """A node that represents the observation in the search tree."""

    def __init__(self, obs, depth=-1):
        self.obs = obs
        self.depth = depth
        self.visit_count = 0
        self.particle_bin = []
        self.children = []

    def find_child_by_e2w(self, tau, epsilon):
        """Randomly returns a child action node according to e2w policy."""
        # assert self.children is not None
        if self.visit_count == 0:
            prob_list = [1 / len(self.children)] * len(self.children)
        else:
            # Get probabilaty list
            lambda_h = np.min([epsilon * len(self.children) /
                               np.log(self.visit_count + 1), 1])
            softmax_value = tau * np.log(np.sum([np.exp(c.mean_value / tau)
                                                 for c in self.children]))
            softmax_policy = [np.exp((c.mean_value - softmax_value) / tau)
                              for c in self.children]
            prob_list = [(1-lambda_h) * sp + lambda_h * 1 /
                         len(self.children) for sp in softmax_policy]
            prob_list /= np.sum(prob_list)

        return np.random.choice(self.children, p=prob_list)

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
