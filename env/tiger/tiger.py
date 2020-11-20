from statistic.step_record import StepRecord

from env.env import Environment
from env.tiger.tiger_world_state import WorldState
from env.tiger.tiger_action import Action
from env.tiger.tiger_obs import Observation

import numpy as np


class Tiger(Environment):
    """Environment class: Tiger

    Define the Tiger environment as a single-player FOG.
    Replace chance nodes with transition probabilities.
    """

    def __init__(self, listen_coefficient=0.75):
        """Init the Tiger class with the listen coefficient cfg."""

        self.name = 'Tiger'
        self.multi_agent = False
        self.listen_coefficient = listen_coefficient

    def new_initial_state(self):
        """Get a new initial world state by initial distribution."""

        return WorldState(np.random.randint(2))
    
    def initial_states(self):
        """Return two lists of the initial states and the probability."""

        states_list = [WorldState(0), WorldState(1), WorldState(-1)]
        prob_list = [1 / len(states_list) for x in states_list]
        return states_list, prob_list
    
    def get_all_states(self):
        """Return a list of all world states."""
        
        return [WorldState(0), WorldState(1), WorldState(-1)]

    def step(self, world_state, action):
        """Get step result by given world state and action."""

        step_record = StepRecord()

        step_record.state = world_state
        step_record.action = action

        # Get observation
        # When listen, get the correct obs with a listen coefficient probability
        if action.encode == 2:  # listen
            r = np.random.rand()
            if r < self.listen_coefficient:
                step_record.obs = Observation(world_state.encode)
            else:
                step_record.obs = Observation(1 - world_state.encode)
        # When open the door, always get the correct obs
        else:
            step_record.obs = Observation(world_state.encode)

        # Get reward
        if action.encode == 2:  # listen
            step_record.reward = -1
        else:
            step_record.reward = 20 if action.encode == world_state.encode \
                else -100

        # Get next world state
        if action.encode == 2:  # listen
            step_record.next_state = WorldState(world_state.encode)
        else:
            step_record.next_state = WorldState(-1)  # terminal
            # The game is over
            step_record.is_terminal = True

        return step_record
