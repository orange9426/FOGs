from util.console import console

module = 'STEPRECORD'


class StepRecord(object):
    """Step record object to store all relevant elements in a step.

    Represents the records of a complete step in the model, including the
    state, action, next state, observations after the action, and reward.
    For convenience, this also includes a boolean flag representing whether
    or not the recording next state is a terminal state.
    """

    def __init__(self, state=None, action=None, next_state=None,
                 obs=None, reward=0, is_terminal=False):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.obs = obs
        self.reward = reward
        self.is_terminal = is_terminal

    def show(self):
        console(3, module, 'State: ' + self.state.to_string())
        console(3, module, 'Action: ' + self.action.to_string())
        if not isinstance(self.obs, tuple):
            console(3, module, 'Observation: ' + self.obs.to_string())
        else:
            console(3, module, 'Observation: ' +
                    '; '.join(o.to_string() for o in self.obs))
        console(3, module, "Reward: " + str(self.reward))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([x == y for x, y in zip(vars(self).values(),
                                               vars(other).values())])
        else:
            return False

    def __ne__(self, other):
        return not self == other
