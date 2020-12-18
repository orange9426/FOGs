import env.environment as e
from env.rock_sample.rock_sample_char import *
from policy.step_record import StepRecord

import numpy as np
import copy


class RockSample(e.Environment):
    """Environment class: Rock Sample

    Define the RockSample environment as a single-player FOG.
    Replace transition probabilities with chance nodes.
    """

    def __init__(self):
        """Init the Rock Sample class."""

        self.name = 'Tiger'
        self.num_players = 1

    def initial_state(self):
        """Get a new initial world state."""

        map_encode = [['.', '-', '.', '.', '.', '.'],
                      ['.', '.', '.', '-', '.', '.'],
                      ['.', '.', '.', '.', '.', '.'],
                      ['.', '-', '.', '.', '.', '.'],
                      ['.', '.', '.', '-', '.', '.']]
        return WorldState(map_encode)

    def initial_obs(self):
        """Get new initial observations."""

        return Observation(-1)  # none

    def step(self, world_state, action):
        """Get step result by given world state and action."""

        step_record = StepRecord()

        step_record.state = world_state
        step_record.action = action

        pos = world_state.pos

        # Chance node
        if world_state.is_chance():
            map_encode = copy.deepcopy(world_state.encode)
            for i, rock_info in enumerate(world_state.rock_list):
                map_encode[rock_info[0][0]][rock_info[0][1]] = 'o' \
                    if action.encode // 2**i % 2 == 1 else 'x'
            map_encode[2][0] = '.p'
            step_record.next_state = WorldState(map_encode)
            step_record.obs = Observation(-1)  # none
            step_record.reward = 0

            return step_record

        # Get next world state
        map_encode = copy.deepcopy(world_state.encode)
        if action.encode == 0:  # east
            map_encode[pos[0]][pos[1]] = map_encode[pos[0]][pos[1]][0]
            map_encode[pos[0]][pos[1] + 1] += 'p'
        elif action.encode == 1:  # south
            map_encode[pos[0]][pos[1]] = map_encode[pos[0]][pos[1]][0]
            map_encode[pos[0] + 1][pos[1]] += 'p'
        elif action.encode == 2:  # west
            map_encode[pos[0]][pos[1]] = map_encode[pos[0]][pos[1]][0]
            map_encode[pos[0]][pos[1] - 1] += 'p'
        elif action.encode == 3:  # north
            map_encode[pos[0]][pos[1]] = map_encode[pos[0]][pos[1]][0]
            map_encode[pos[0] - 1][pos[1]] += 'p'
        elif action.encode == 4:  # sample
            map_encode[pos[0]][pos[1]] = '-p'
        step_record.next_state = WorldState(map_encode)

        # Get observation
        if action.encode == 4:  # sample
            step_record.obs = Observation(1) if world_state.map[pos[0]][pos[1]] \
                is 'o' else Observation(0)
        elif action.encode > 4:  # check
            rock_info = world_state.rock_list[action.encode - 5]
            distance = np.abs(rock_info[0][0] - pos[0]) + \
                np.abs(rock_info[0][1] - pos[1])
            if np.random.rand() < np.math.pow(2, -distance / 10):
                step_record.obs = Observation(1) \
                    if rock_info[1] is 'good' else Observation(0)
            else:
                step_record.obs = Observation(1) \
                    if rock_info[1] is 'bad' else Observation(0)
        else:
            step_record.obs = Observation(-1)  # none

        # Get reward
        if action.encode == 4:  # sample
            step_record.reward = 20 if world_state.map[pos[0]][pos[1]] \
                is 'o' else -20
        elif action.encode > 4:  # check
            step_record.reward = 0
        else:
            step_record.reward = 0

        step_record.is_terminal = step_record.next_state.is_terminal()

        return step_record

    def possible_states(self, obs):
        """Return a list of all possilble states corresponding to current obs."""

        inital_state = self.initial_state()
        possible_states = [self.step(inital_state, action).next_state
                           for action in inital_state.legal_actions()]
        prob_list = [1 / len(possible_states)
                     for _ in range(len(possible_states))]
        return possible_states, prob_list

    def preferred_rollout_policy(self, state, obs):
        """Rollout policy with human knowledge."""

        action = np.random.choice(state.legal_actions())

        return action
