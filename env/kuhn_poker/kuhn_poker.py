from statistic.step_record import StepRecord
from statistic.record_history import History

from env.env import Environment
from env.kuhn_poker.kuhn_poker_world_state import WorldState
from env.kuhn_poker.kuhn_poker_action import Action
from env.kuhn_poker.kuhn_poker_obs import PrivateObservation
from env.kuhn_poker.kuhn_poker_obs import PublicObservation

import numpy as np
import copy


class KuhnPoker(Environment):
    """Environment class: Kuhn Poker

    Define the Kuhn Poker environment as a FOG.
    Replace transition probabilities with chance nodes.
    """

    def __init__(self):
        """Init Kuhn Poker class with naming."""

        self.name = 'Kuhn Poker'
        self.multi_agent = True

    def initial_state(self):
        """Get a new initial world state."""

        return WorldState([[-1, -1], [1, 1], -1])

    def initial_obs(self):
        """Get new initial observations."""

        return (PrivateObservation(-1, player=0), PrivateObservation(-1, player=0),
                PublicObservation([1, 1]))

    def initial_history(self):
        """Get new initial history with the initial record."""

        return History([StepRecord(next_state=self.initial_state(),
                                   obs=self.initial_obs())], self)

    def get_all_histories(self):
        """Return a list of all possible histories in the game."""

        history_list = []
        history_queue = [self.initial_history()]

        while history_queue:
            history = history_queue.pop(0)
            history_list.append(history)
            if not history[-1].is_terminal:
                history_queue += [history.child(action) for action in
                                  history[-1].next_state.legal_actions()]

        return history_list

    def step(self, world_state, action):
        """Get the step result given a world state and an action."""

        step_record = StepRecord()

        step_record.state = world_state
        step_record.action = action

        # Get next world state and rewards
        w_encode = copy.deepcopy(world_state.encode)
        reward = 0
        if world_state.is_chance():  # is chance
            w_encode[0] = copy.deepcopy(action.encode)
            w_encode[-1] = 0  # player 1's turn
        else:  # pass
            if action.encode == 0:  # pass
                # The game will not end after 'pass' only if at the beginning
                if w_encode[1] == [1, 1] and w_encode[-1] == 0:
                    w_encode[-1] = 1 - w_encode[-1]  # opponent's turn
                else:  # the game is over
                    w_encode[-1] = -2  # terminal
                    step_record.is_terminal = True
                    if w_encode[1] == [1, 1]:  # all pass
                        reward = 1 if w_encode[0][0] > w_encode[0][1] else -1
                    else:
                        # The player who passes will lose
                        reward = 1 if action.player == 1 else -1
            else:  # bet
                w_encode[1][action.player] += 1  # the bet of current player +1
                # The game will end after 'bet' only if all bet
                if w_encode[1] == [2, 2]:
                    w_encode[-1] = -2  # terminal
                    step_record.is_terminal = True
                    reward = 2 if w_encode[0][0] > w_encode[0][1] else -2
                else:
                    w_encode[-1] = 1 - w_encode[-1]  # opponent's turn
        step_record.next_state = WorldState(w_encode)
        step_record.reward = reward

        # Get observation
        # Observations always match the world state
        step_record.obs = (PrivateObservation(w_encode[0][0], 0),
                           PrivateObservation(w_encode[0][1], 1),
                           PublicObservation(w_encode[1]))

        return step_record
