import env.environment as e
from env.texas_holdem.texas_holdem_char import *
from env.public_belief_state import PublicBeliefState
from util.step_record import StepRecord

import numpy as np
import copy
import torch

# Phase
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

# Player
CHANCE = -1
TERMINAL = -2

# Status
FOLDED = -2
NONRESPONSE = -1
CALLED = 0
RAISED1TIME = 1
RAISED2TIMES = 2


class TexasHoldem(e.Environment):
    """Environment class: Texas Holdem."""

    def __init__(self):
        """Init Texas Holdem class."""

        self.name = 'Texas Holdem'
        self.num_players = 2

    def initial_state(self):
        """Get a new initial world state."""

        if not hasattr(self, '_initial_state'):
            self._initial_state = WorldState(
                hand=[[], []], pot=[2, 1], pub=[], phase=PREFLOP,
                status=[NONRESPONSE, NONRESPONSE], player=CHANCE
            )

        return self._initial_state

    def initial_obs(self):
        """Get new initial observations."""

        if not hasattr(self, '_initial_obs'):
            self._initial_obs = (
                PrivateObservation(hand=[], player=0),
                PrivateObservation(hand=[], player=1),
                PublicObservation(pot=[2, 1], pub=[])
            )

        return self._initial_obs

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

    def step(self, state, action):
        """Get the step result given a world state and an action."""

        assert not state.is_terminal()

        # Get next world state
        next_state = copy.deepcopy(state)
        # Is chance
        if state.is_chance():
            if state.phase == PREFLOP:  # preflop
                next_state.hand[0] = action.deal[:2]
                next_state.hand[1] = action.deal[2:]
                next_state.player = 1  # player 2's turn
            elif state.phase == FLOP:  # flop
                next_state.pub += action.deal
                next_state.player = 0  # player 1's turn
            elif state.phase == TURN:  # turn
                next_state.pub += action.deal
                next_state.player = 0  # player 1's turn
            else:  # river
                next_state.pub += action.deal
                next_state.player = 0  # player 1's turn
        # Is player
        else:
            current_player = state.player  # current player
            # Choose fold
            if action.action == FOLD:
                next_state.status[current_player] = FOLDED
                next_state.player = TERMINAL  # game over
            # Choose call
            elif action.action == CALL:
                next_state.pot[current_player] = state.pot[1-current_player]
                if state.status[1 - current_player] != NONRESPONSE:  # phase finish
                    next_state.status = [NONRESPONSE, NONRESPONSE]
                    if state.phase == RIVER:  # last phase
                        next_state.player = TERMINAL  # game over
                    else:  # turn to next phase
                        next_state.phase += 1
                        next_state.player = CHANCE
                else:  # phase will not finish
                    next_state.status[current_player] = CALLED
                    next_state.player = 1 - current_player  # opponent's turn
            # Choose raise
            else:
                next_state.pot[current_player] = \
                    state.pot[1-current_player] + action.bet
                if state.status[current_player] == NONRESPONSE:
                    next_state.status[current_player] = CALLED
                next_state.status[current_player] += 1  # raise times + 1
                next_state.player = 1 - current_player

        # Get observation
        obs = (PrivateObservation(hand=next_state.hand[0], player=0),
               PrivateObservation(hand=next_state.hand[1], player=1),
               PublicObservation(pot=next_state.pot, pub=next_state.pub))

        # Get reward
        if not next_state.is_terminal():
            reward = 0
        else:
            if next_state.winner == -1:  # draw
                reward = 0
            elif next_state.winner == 0:
                reward = next_state.pot[1]
            else:
                reward = -next_state.pot[0]

        return StepRecord(state, action, next_state, obs, reward)

    def get_tensor(self, pbs):
        """Get the tensor of a public belief state."""

        state = pbs.history_list[0][-1].next_state
        # Get tensor such like [round, bet1, bet2, pub_hand, turn, *prob_dict]
        pbs_list = [*state.pot, *state.pub,
                    state.phase, *state.status, state.player,
                    *[prob for prob in pbs.prob_dict.values()]]
        pbs_tensor = torch.tensor(pbs_list)

        return pbs_tensor
