import env.environment as e
from env.leduc_poker.leduc_poker_char import *
from util.step_record import StepRecord
from env.public_belief_state import PublicBeliefState

import numpy as np
import copy
import torch


class LeducPoker(e.Environment):
    """Environment class: Leduc Poker

    Define the Leduc Poker environment as a FOG.
    Replace transition probabilities with chance nodes.
    """

    def __init__(self):
        """Init Leduc Poker class."""

        self.name = 'LeducPoker'
        self.num_players = 2

    def initial_state(self):
        """Get a new initial world state."""

        return WorldState([[-1, -1, -1], [1, 1], -1])

    def initial_obs(self):
        """Get new initial observations."""

        return (PrivateObservation(-1, player=0), PrivateObservation(-1, player=1),
                PublicObservation([[1, 1], -1]))

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

        # Get next world state
        w_encode = copy.deepcopy(world_state.encode)
        if world_state.is_chance():  # is chance
            if world_state.phase == 0:  # private hands chance
                w_encode[0][0] = action.encode[0]
                w_encode[0][1] = action.encode[1]
            else:  # public hands chance
                w_encode[0][2] = action.encode
            w_encode[-1] = 0  # player 1's turn
        else:
            if action.encode == 0:  # pass
                # The game will end after 'pass' if one's bet are more
                #   or player 2 passes in phase 2
                if world_state.bet[0] != world_state.bet[1] or \
                        (world_state.player == 1 and world_state.phase == 1):
                    w_encode[-1] = -2  # terminal
                # The game will turn to chance when player 2 passes in phase 1
                elif world_state.player == 1 and world_state.phase == 0:
                    w_encode[-1] = -1  # chance
                else:
                    w_encode[-1] = 1 - w_encode[-1]  # opponent's turn
            else:  # bet
                w_encode[1][action.player] *= 2  # the bet of current player *2
                # The game will end after 'bet' if two players have equal bets in phase 2
                #   and will turn to chance if in phase 1
                if w_encode[1][0] == w_encode[1][1]:
                    if world_state.phase == 1:
                        w_encode[-1] = -2  # terminal
                    elif world_state.phase == 0:
                        w_encode[-1] = -1  # chance
                else:
                    w_encode[-1] = 1 - w_encode[-1]  # opponent's turn
        next_world_state = WorldState(w_encode)

        # Get observation
        # Observations always match the world state
        obs = (PrivateObservation(next_world_state.hand[0], 0),
               PrivateObservation(next_world_state.hand[1], 1),
               PublicObservation([next_world_state.bet, next_world_state.pub]))

        # Get reward
        if not next_world_state.is_terminal():
            reward = 0
        else:
            if next_world_state.winner == -1:  # draw
                reward = 0
            elif next_world_state.winner == 0:
                reward = next_world_state.bet[1 - next_world_state.winner]
            else:
                reward = -next_world_state.bet[1 - next_world_state.winner]

        return StepRecord(world_state, action, next_world_state, obs, reward)

    def get_tensor(self, pbs):
        """Get the tensor of a public belief state such like
        [round, bet1, bet2, pub_hand, turn, *prob_dict]."""

        public_state = pbs.public_state
        # Get tensor such like [round, bet1, bet2, pub_hand, turn, *prob_dict]
        pbs_list = [len(public_state), public_state[-1].bet[0], public_state[-1].bet[1],
                    public_state[-1].pub, pbs.current_player(),
                    *[prob for prob in pbs.prob_dict.values()]]
        pbs_tensor = torch.tensor(pbs_list)

        return pbs_tensor
