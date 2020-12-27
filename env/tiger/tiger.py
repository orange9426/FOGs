import env.environment as e
from env.tiger.tiger_char import *
from util.step_record import StepRecord

import numpy as np


class Tiger(e.Environment):
    """Environment class: Tiger

    Define the Tiger environment as a single-player FOG.
    Replace transition probabilities with chance nodes.
    """

    def __init__(self, listen_coefficient=0.75):
        """Init the Tiger class with the listen coefficient cfg."""

        self.name = 'Tiger'
        self.num_players = 1
        self.listen_coefficient = listen_coefficient

    def initial_state(self):
        """Get a new initial world state."""

        return WorldState(-1)  # chance

    def initial_obs(self):
        """Get new initial observations."""

        return Observation(-1)  # none

    def step(self, world_state, action):
        """Get step result by given world state and action."""

        step_record = StepRecord()

        step_record.state = world_state
        step_record.action = action

        # Chance node
        if world_state.is_chance():
            step_record.next_state = WorldState(action.encode)
            step_record.obs = Observation(-1)  # none
            step_record.reward = 0

            return step_record

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
            step_record.next_state = WorldState(-2)  # terminal

        return step_record
    
    def possible_states(self, obs):
        """Return a list of all possilble states corresponding to initial obs."""

        return [WorldState(0), WorldState(1)], [0.5, 0.5]
