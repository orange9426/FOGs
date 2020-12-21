class PublicBeliefState(object):
    """Public Belief State object introducted in ReBeL.

    It contains a public state and the probability distribution of all possible
    information states of all players conforming to that public state.
    """

    def __init__(self, public_state, prob_dict):
        self.public_state = public_state
        self.prob_dict = prob_dict

    def child(self, action, policy):
        """Get the child PBS given the action and policy."""

        # Set the child public state
        public_state = self.get_all_histories()[0].child(action).get_public_state()

        # Set the child prob dict
        prob_dict = {}
        total = 0
        for history in self.get_all_histories():
            child_history = history.child(action)
            prob_dict[child_history.to_string()] = self.prob_dict[history.to_string()] \
                * policy.get_prob_dict(history)[action]  # TODO: policy.get_prob_dict()
            total += prob_dict[child_history.to_string()]
        prob_dict = {k: v / total for k, v in prob_dict.items()}  # normalization

        return PublicBeliefState(public_state, prob_dict)

    def get_all_histories(self):
        """Given a list of all histories, get a list of all possible histories
        corresponding to the public state."""

        if not hasattr(self, '_history_list'):
            self._history_list = self.public_state.get_all_histories()

        return self._history_list
