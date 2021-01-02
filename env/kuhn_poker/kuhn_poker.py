import env.environment as e
from env.kuhn_poker.kuhn_poker_char import *
from util.step_record import StepRecord
from env.public_belief_state import PublicBeliefState

import numpy as np
import copy
import torch


class KuhnPoker(e.Environment):
    """Environment class: Kuhn Poker

    Define the Kuhn Poker environment as a FOG.
    Replace transition probabilities with chance nodes.
    """

    def __init__(self):
        """Init Kuhn Poker class."""

        self.name = 'KuhnPoker'
        self.num_players = 2

    def initial_state(self):
        """Get a new initial world state."""

        return WorldState([[-1, -1], [1, 1], -1])

    def initial_obs(self):
        """Get new initial observations."""

        return (PrivateObservation(-1, player=0), PrivateObservation(-1, player=1),
                PublicObservation([1, 1]))

    def initial_pbs(self):
        """Get new initial public belief state."""

        if not hasattr(self, '_initial_pbs'):
            initial_history = self.initial_history()
            public_state = initial_history.child(
                initial_history.legal_actions()[0]).get_public_state()
            prob_dict = {initial_history.child(a).to_string(): p for
                         a, p in zip(*initial_history.chance_outcomes())}
            self._initial_pbs = PublicBeliefState(public_state, prob_dict)

        return self._initial_pbs

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
        else:
            if action.encode == 0:  # pass
                # The game will not end after 'pass' only if at the beginning
                if w_encode[1] == [1, 1] and w_encode[-1] == 0:
                    w_encode[-1] = 1 - w_encode[-1]  # opponent's turn
                else:  # the game is over
                    w_encode[-1] = -2  # terminal
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

    def get_tensor(self, pbs):
        """Get the tensor of a public belief state such like
        [round, bet1, bet2, *prob_dict]."""

        public_state = pbs.public_state
        # Get tensor such like [round, bet1, bet2, *prob_dict]
        pbs_list = [len(public_state), public_state[-1].encode[0],
                    public_state[-1].encode[1],
                    *[prob for prob in pbs.prob_dict.values()]]
        pbs_tensor = torch.tensor(pbs_list)

        return pbs_tensor
