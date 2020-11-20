from statistic.record_history import History
from statistic.step_record import StepRecord

from solver.solver import Solver
from solver.me_pomcp.obs_node import ObservationNode
from solver.me_pomcp.action_node import ActionNode

from util.console import console
from util.divider import print_divider

import numpy as np

module = "ME-POMCP"


class ME_POMCP(Solver):
    """
    Solver: ME-POMCP
    """

    def __init__(self, args):
        # Name of the solver
        self.name = args['solver']

        # Discount factor of the model
        self.discount = args['discount']

        # Flag of whether to print step messages
        self.quiet = args['quiet']

        # Num of MC sims to do at each belief node
        self.n_sims = args['n_sims']

        # Num of state particles to generate for root
        self.n_start_states = args['n_start_states']

        # Lower bound on num of particles a belief
        self.min_particle_count = args['min_particle_count']
        # Upper bound on num of particles a belief
        self.max_particle_count = args['max_particle_count']

        # Max depth for a DFS of the belief search tree in MCTS
        self.max_depth = args['max_depth']

        # Parameter tau of softmax policies
        self.me_tau = args['me_tau']

        # Parameter epsilon of softmax policies
        self.me_epsilon = args['me_epsilon']

        # Function pointer
        self.evaluation_fn = self._rollout
        self.rollout_policy = self._random_policy

    def reset_for_epoch(self):
        """Initial the solver before solving the game."""
        pass

    def solve_game(self, env):
        """Solve the entire game for one epoch."""

        state = env.new_initial_state()
        obs = state.initial_obs()

        # Set the root node and the corresponding particle bin
        root = ObservationNode(obs, depth=0)
        for _ in range(self.n_start_states):
            particle = env.new_initial_state()
            while particle.initial_obs() != obs:
                particle = env.new_initial_state()
            root.particle_bin.append(particle)

        history = History()

        # Solve the game by step until a terminal state
        while not state.is_terminal():
            # Get an action by planning
            action = self._solve_one_step(root, env)

            # Get step result
            step_record = env.step(state, action)

            # Show the step
            if not self.quiet:
                print_divider('small')
                console(3, module, "Step: " + str(root.depth))
                step_record.show()

            history.append(step_record)

            state = step_record.next_state
            root = root.find_child(action).find_child(step_record.obs)

        return history

    def _solve_one_step(self, root, env):
        """Solve and return an action at some state."""

        # Do simulations for n times
        for _ in range(self.n_sims):
            # Sample an initial state for a simulation
            state = np.random.choice(root.particle_bin)

            # Selection and Expansion
            visit_path, record_history, working_state = self._apply_tree_policy(
                state, root, env)

            # Evaluation
            ev_return = self.evaluation_fn(working_state, env)

            # Back up
            for action_node, obs_node in reversed(visit_path):
                step_record = record_history.pop()

                obs_node.visit_count += 1
                obs_node.particle_bin.append(step_record.next_state)

                action_node.visit_count += 1
                if not obs_node.children:
                    update_value = step_record.reward + self.discount * ev_return
                else:
                    softmax_value = self.me_tau * np.log(
                        np.sum([
                            np.exp(c.mean_value / self.me_tau)
                            for c in obs_node.children
                        ]))
                    update_value = step_record.reward + self.discount + softmax_value
                action_node.total_reward += update_value

            root.visit_count += 1

        return root.best_child().action

    def _apply_tree_policy(self, state, root, env):
        """Select nodes according to the tree policy in the search tree."""

        visit_path = []
        record_history = History()
        working_state = state
        current_node = root
        depth = root.depth
        # Select in the tree until a new node or a terminal node or reaching the max depth
        while current_node.visit_count > 0 and not working_state.is_terminal() \
                and depth <= root.depth + self.max_depth:
            # For a new node, initialize its children, then choose a child as normal
            if not current_node.children:
                legal_actions = working_state.legal_actions()
                # Reduce bias from move generation order.
                np.random.shuffle(legal_actions)
                current_node.children = [
                    ActionNode(action, depth) for action in legal_actions
                ]

            # Choose a child by e2w policy
            action_child = current_node.find_child_by_e2w(
                self.me_tau, self.me_epsilon)

            # Get step result and turn to the action child node
            step_record = env.step(working_state, action_child.action)
            current_node = action_child
            depth += 1

            # Turn to the obs child node, if not exists, append a new node
            obs_child = current_node.find_child(step_record.obs)
            if not obs_child:
                obs_child = ObservationNode(step_record.obs, depth)
                current_node.children.append(obs_child)

            current_node = obs_child
            working_state = step_record.next_state

            # Add node to visit path and return it
            visit_path.append((action_child, obs_child))
            record_history.append(step_record)

        return visit_path, record_history, working_state

    def _rollout(self, state, env):
        """Rollout method to evaluate a state."""

        history = History()

        # Rollout to terminal state and return the discounted reward
        while not state.is_terminal():
            action = self.rollout_policy(state)
            step_record = env.step(state, action)
            state = step_record.next_state

            history.append(step_record)

        return history.discounted_return(self.discount)

    def _random_policy(self, state):
        """Random policy."""
        return np.random.choice(state.legal_actions())
