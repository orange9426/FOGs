from policy.policy import TabularPolicy
from policy.policy import TabularPolicy_Subgame
from solver.solver import Solver
from env.public_belief_state import PublicBeliefState

import torch
import numpy as np
import collections
import attr


@attr.s
class InfoStateNode(object):
    legal_actions = attr.ib()
    index_in_tabular_policy = attr.ib()
    cumulative_regret = attr.ib(factory=lambda: collections.defaultdict(float))
    cumulative_policy = attr.ib(factory=lambda: collections.defaultdict(float))


def _update_current_policy(current_policy, info_state_nodes):
    for info_state, info_state_node in info_state_nodes.items():
        info_state_policy = current_policy.policy_for_key(info_state)

        for action, value in _regret_matching(info_state_node.cumulative_regret,
                                              info_state_node.legal_actions).items():
            info_state_policy[action] = value


def _regret_matching(cumulative_regrets, legal_actions):
    regrets = cumulative_regrets.values()
    sum_positive_regrets = sum((regret for regret in regrets if regret > 0))
    info_state_policy = {}
    if sum_positive_regrets > 0:
        for i in range(len(legal_actions)):
            positive_action_regret = max(0.0, cumulative_regrets[i])
            info_state_policy[i] = positive_action_regret / \
                sum_positive_regrets
    else:
        for i in range(len(legal_actions)):
            info_state_policy[i] = 1.0 / len(legal_actions)
    return info_state_policy


def _update_average_policy(average_policy, info_state_nodes):
    for info_state, info_state_node in info_state_nodes.items():
        info_state_policies_sum = info_state_node.cumulative_policy
        info_state_policy = average_policy.policy_for_key(info_state)
        probabilities_sum = sum(info_state_policies_sum.values())
        if probabilities_sum == 0:
            num_actions = len(info_state_policy)
            for i in range(num_actions):
                info_state_policy[i] = 1 / num_actions
        else:
            for action, action_prob_sum in info_state_policies_sum.items():
                info_state_policy[action] = action_prob_sum / probabilities_sum


class CFR(Solver):
    """
    Solver: CFR
    """
    online = False

    def __init__(self, game, args):
        self._game = game
        self.name = args['solver']
        self.iterations = args['n_epochs']
        self._num_players = self._game.num_players
        self._root_node = self._game.initial_history()  # !!!
        self._current_policy = TabularPolicy(self._game)
        self._average_policy = self._current_policy.__copy__()

        self._info_state_nodes = {}
        self._initialize_info_states_nodes(self._root_node)

    def _initialize_info_states_nodes(self, history):
        if history.is_terminal():
            return

        if history.is_chance():
            for action in history.legal_actions():
                self._initialize_info_states_nodes(history.child(action))
            return

        current_player = history.current_player()
        info_state = history.get_info_state()[current_player].to_string()

        info_state_node = self._info_state_nodes.get(info_state)
        if info_state_node is None:
            legal_actions = history.legal_actions()
            info_state_node = InfoStateNode(
                legal_actions=legal_actions,
                index_in_tabular_policy=self._current_policy.history_lookup[info_state]
            )
            self._info_state_nodes[info_state] = info_state_node

        for action in info_state_node.legal_actions:
            self._initialize_info_states_nodes(history.child(action))

    def current_policy(self):
        return self._current_policy

    def average_policy(self):
        _update_average_policy(self._average_policy, self._info_state_nodes)
        return self._average_policy

    def _compute_counterfactual_regret_for_player(self, history, reach_probabilities, player):
        if history.is_terminal():
            return np.asarray([history.get_return(), -history.get_return()])

        if history.is_chance():
            history_value = 0.0
            for action, action_prob in zip(*history.chance_outcomes()):
                new_history = history.child(action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob
                history_value += action_prob * self._compute_counterfactual_regret_for_player(
                    new_history, new_reach_probabilities, player)
            return history_value

        current_player = history.current_player()
        info_state = history.get_info_state()[current_player].to_string()

        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        history_value = np.zeros(self._num_players)
        children_utilities = {}
        info_state_node = self._info_state_nodes[info_state]
        info_state_policy = self._current_policy.action_probabilities_table[
            info_state_node.index_in_tabular_policy
        ]

        legal_actions = history.legal_actions()

        for i in range(len(legal_actions)):
            action_prob = info_state_policy[i]
            new_history = history.child(legal_actions[i])
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= action_prob
            child_utility = self._compute_counterfactual_regret_for_player(
                new_history, new_reach_probabilities, player=player
            )
            history_value += action_prob * child_utility
            children_utilities[i] = child_utility

        if current_player != player:
            return history_value

        reach_prob = reach_probabilities[current_player]
        counterfactual_reach_prob = (
            np.prod(reach_probabilities[:current_player]) *
            np.prod(reach_probabilities[current_player+1:])
        )
        history_value_for_player = history_value[current_player]
        for i in range(len(legal_actions)):
            cfr_regret = counterfactual_reach_prob * (
                children_utilities[i][current_player] -
                history_value_for_player
            )
            info_state_node.cumulative_regret[i] += cfr_regret
            info_state_node.cumulative_policy[i] += reach_prob * \
                info_state_policy[i]
        return history_value

    def evaluate_and_update_policy(self):
        for player in range(self._num_players):
            self._compute_counterfactual_regret_for_player(
                self._root_node,
                reach_probabilities=np.ones(self._num_players+1),
                player=player
            )
            _update_current_policy(self._current_policy,
                                   self._info_state_nodes)

            # self.print_policy(self.current_policy())

    def print_policy(self, policy):
        policy_dict = {
            key: policy.action_probabilities_table[policy.history_lookup[key]]
            for key in policy.history_lookup.keys()}
        print(policy_dict)

    def reset_for_epoch(self):
        """Initialize the solver before solving the game."""
        self._current_policy = TabularPolicy(self._game)
        self._average_policy = self._current_policy.__copy__()

    def train_policy(self):
        """Solve the entire game for one epoch."""
        self.iterations = 10
        # print(self.current_policy().action_probabilities_table)
        for i in range(self.iterations):
            self.evaluate_and_update_policy()
        print(self.average_policy().action_probabilities_table)
        return self._average_policy


class DepthLimited_CFR(Solver):
    """
    Solver: DepthLimited_CFR
    """
    online = False

    def __init__(self, game, net, pbs, max_depth, iteration_num):
        self._game = game
        # self.name = args['solver']
        # self.iterations = args['n_epochs']
        self._num_players = self._game.num_players
        self.max_depth = max_depth
        self.initial_pbs = pbs
        self._root_pbs = self.initial_pbs  # !!!
        self._current_policy = TabularPolicy_Subgame(
            self._game, pbs, max_depth)
        self._average_policy = self._current_policy.__copy__()
        self.value_net = net
        self.values_dict = {}
        self.iteration_num = iteration_num

        self._info_state_nodes = {}
        self._initialize_info_states_nodes(self._root_pbs)

    def _initialize_info_states_nodes(self, node):
        if isinstance(node, PublicBeliefState):
            for history in node.history_list:
                self._initialize_info_states_nodes(history)
        else:
            history = node

            if history.is_terminal():
                return

            if self._current_policy.leaf_dict[history.to_string()]:
                return

            if history.is_chance():
                for action in history.legal_actions():
                    self._initialize_info_states_nodes(history.child(action))
                return

            current_player = history.current_player()
            info_state = history.get_info_state()[current_player].to_string()

            info_state_node = self._info_state_nodes.get(info_state)
            if info_state_node is None:
                legal_actions = history.legal_actions()
                info_state_node = InfoStateNode(
                    legal_actions=legal_actions,
                    index_in_tabular_policy=self._current_policy.history_lookup[info_state]
                )
                self._info_state_nodes[info_state] = info_state_node

            for action in info_state_node.legal_actions:
                self._initialize_info_states_nodes(history.child(action))

    def current_policy(self):
        return self._current_policy

    def average_policy(self):
        _update_average_policy(self._average_policy, self._info_state_nodes)
        return self._average_policy

    def set_leaf_values(self, pbs):
        history = pbs.history_list[0]
        if self._current_policy.leaf_dict[history.to_string()]:  # is leaf
            self.values_dict[pbs.public_state.to_string()] = self.value_net(
                pbs.to_tensor()).tolist()  # !
        else:
            for action in pbs.legal_actions():
                self.set_leaf_values(pbs.child(action, self._current_policy))

    def get_training_data(self):
        self._current_policy = self.average_policy().__copy__()  # TODO:?
        label = []
        player = 0
        for history in self.initial_pbs.history_list:
            reach = np.ones(self._num_players+1)
            reach[-1] = self.initial_pbs.prob_dict[history.to_string()]
            value = self._compute_counterfactual_regret_for_player(
                history,
                reach_probabilities=reach,
                player=player
            )[0]
            label.append(value)
        label = torch.tensor(label, dtype=torch.float32)
        return (self.initial_pbs.to_tensor(), label)

    def sample_pbs(self):
        initial_prob = []
        initial_history = []
        for history in self.initial_pbs.history_list:
            prob = self.initial_pbs.prob_dict[history.to_string()]
            initial_prob.append(prob)
            initial_history.append(history)
        index = np.random.choice(np.arange(len(initial_prob)), p=initial_prob)
        history = initial_history[index]
        random_player = np.random.randint(self._num_players)
        action_list = []
        while not history.is_terminal() and not self._current_policy.leaf_dict[history.to_string()]:
            if history.current_player == random_player:
                i = np.random.randint(len(history.legal_actions()))
                action = history.legal_actions()[i]
                history = history.child(action)
            elif history.is_chance():
                action = np.random.choice(history.chance_outcomes()[
                                          0], p=history.chance_outcomes()[1])
                history = history.child(action)
            else:
                info_state = history.get_info_state(
                )[history.current_player()].to_string()
                policy = self._average_policy.policy_for_key(info_state)
                i = np.random.choice(np.arange(len(policy)), p=policy)
                action = history.legal_actions()[i]
                history = history.child(action)
            action_list.append(action)
        pbs = self.initial_pbs
        for action in action_list:
            pbs = pbs.child(action, self._average_policy)
        return pbs

    def _compute_counterfactual_regret_for_player(self, history, reach_probabilities, player):
        if history.is_terminal():
            return np.asarray([history.get_return(), -history.get_return()])

        if self._current_policy.leaf_dict[history.to_string()]:
            pub_s = history.get_public_state()
            u = self.values_dict[pub_s.to_string()][pub_s.get_all_histories().index(history)]
            return np.asarray([u, -u])

        if history.is_chance():
            history_value = 0.0
            for action, action_prob in zip(*history.chance_outcomes()):
                new_history = history.child(action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob
                history_value += action_prob * self._compute_counterfactual_regret_for_player(
                    new_history, new_reach_probabilities, player)
            return history_value

        current_player = history.current_player()
        info_state = history.get_info_state()[current_player].to_string()

        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        history_value = np.zeros(self._num_players)
        children_utilities = {}
        info_state_node = self._info_state_nodes[info_state]
        info_state_policy = self._current_policy.action_probabilities_table[
            info_state_node.index_in_tabular_policy
        ]

        legal_actions = history.legal_actions()

        for i in range(len(legal_actions)):
            action_prob = info_state_policy[i]
            new_history = history.child(legal_actions[i])
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= action_prob
            child_utility = self._compute_counterfactual_regret_for_player(
                new_history, new_reach_probabilities, player=player
            )
            history_value += action_prob * child_utility
            children_utilities[i] = child_utility

        if current_player != player:
            return history_value

        reach_prob = reach_probabilities[current_player]
        counterfactual_reach_prob = (
            np.prod(reach_probabilities[:current_player]) *
            np.prod(reach_probabilities[current_player+1:])
        )
        history_value_for_player = history_value[current_player]
        for i in range(len(legal_actions)):
            cfr_regret = counterfactual_reach_prob * (
                children_utilities[i][current_player] -
                history_value_for_player
            )
            info_state_node.cumulative_regret[i] += cfr_regret
            info_state_node.cumulative_policy[i] += reach_prob * \
                info_state_policy[i]
        return history_value

    def evaluate_and_update_policy(self):
        for player in range(self._num_players):
            for history in self.initial_pbs.history_list:
                reach = np.ones(self._num_players+1)
                reach[-1] = self.initial_pbs.prob_dict[history.to_string()]
                self._compute_counterfactual_regret_for_player(
                    history,
                    reach_probabilities=reach,
                    player=player
                )
            _update_current_policy(self._current_policy,
                                   self._info_state_nodes)
        # for player in range(self._num_players):
        #     self._compute_counterfactual_regret_for_player(
        #         self._root_node,
        #         reach_probabilities=np.ones(self._num_players+1),
        #         player=player
        #     )
        #     _update_current_policy(self._current_policy,
        #                            self._info_state_nodes)

            # self.print_policy(self.current_policy())

    def print_policy(self, policy):
        policy_dict = {
            key: policy.action_probabilities_table[policy.history_lookup[key]]
            for key in policy.history_lookup.keys()}
        print(policy_dict)

    def reset_for_epoch(self):
        """Initialize the solver before solving the game."""
        self._current_policy = TabularPolicy(self._game)
        self._average_policy = self._current_policy.__copy__()

    def train_policy(self):
        """Solve the entire game for one epoch."""
        for i in range(self.iteration_num):
            self.set_leaf_values(self.initial_pbs)
            self.evaluate_and_update_policy()

        return self.average_policy()
