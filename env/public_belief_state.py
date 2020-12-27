class PublicBeliefState(object):
    """Public Belief State object introducted in ReBeL.

    It contains a public state and the probability distribution of all possible
    information states of all players conforming to that public state.
    """

    def __init__(self, public_state, prob_dict):
        self.public_state = public_state
        self.prob_dict = prob_dict

    def is_chance(self):
        """Whether is the chance node."""

        return self.history_list[0].is_chance()

    def is_terminal(self):
        """Whether is the terminal node."""

        return self.history_list[0].is_terminal()

    def current_player(self):
        """Get the current player of the history."""

        return self.history_list[0].current_player()
    
    def legal_actions(self):

        return self.history_list[0].legal_actions()

    def child(self, action, policy):
        """Get the child PBS given the action and policy."""

        # Set the child public state
        public_state = self.history_list[0].child(action).get_public_state()

        # Set the child prob dict
        prob_dict = {}
        total = 0
        for history in self.history_list:
            child_history = history.child(action)
            prob_dict[child_history.to_string()] = self.prob_dict[history.to_string()] \
                * policy.get_prob(history, action)  # TODO: policy.get_prob_dict()
            total += prob_dict[child_history.to_string()]
        prob_dict = {k: v / total for k, v in prob_dict.items()}  # normalization

        return PublicBeliefState(public_state, prob_dict)

    @property
    def history_list(self):
        """Given a list of all histories, get a list of all possible histories
        corresponding to the public state."""

        if not hasattr(self, '_history_list'):
            self._history_list = self.public_state.get_all_histories()

        return self._history_list
