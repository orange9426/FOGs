class Policy(object):
    def __init__(self, game, player_ids):
        self.game = game
        self.player_ids = player_ids

    def action_probabilities(self, history, player_id=None):
        raise NotImplementedError()

    def __call__(self, history, player_id=None):
        return self.action_probabilities(history, player_id)

class TabularPolicy(Policy):
    def __init__(self, game):
        all_players = list(range(game.num_players))
        super(TabularPolicy, self).__init__(game, all_players)
        histories = game.get_all_histories() #!!!

        self.history_lookup = {}
        self.info_state_per_player = [[] for _ in all_players]
        self.legal_actions_list = []
        

        for player in all_players:
            for history in histories:
                if player == history[-1].next_state.current_player():
                    legal_actions = history[-1].next_state.legal_actions()
                    if len(legal_actions):
                        key = self._history_key(history, player)
                        if key not in self.history_lookup:
                            history_index = len(self.legal_actions_list)
                            self.history_lookup[key] = history_index
                            self.legal_actions_list.append(legal_actions)
                            self.info_state_per_player[player].append(key)

        self.action_probabilities_table = []
        for legal_actions in self.legal_actions_list:
            self.action_probabilities_table.append([1/len(legal_actions)]*len(legal_actions))
    
    def _history_key(self, history, player):
        return history.get_info_state()[player].to_string()

    def policy_for_key(self, key):
        policy_index = self.history_lookup[key]
        return self.action_probabilities_table[policy_index]

    def action_probabilities(self, history, player_id):
        policy_index = self.history_lookup[self._history_key(history,history[-1].next_state.current_player())]
        policy = self.action_probabilities_table[policy_index]
        legal_actions = legal_actions_list[policy_index]
        
        return {
            legal_actions[i]: policy[i]
            for i in range(len(legal_actions))
        }

    def __copy__(self):
        result = TabularPolicy.__new__(TabularPolicy)
        result.history_lookup = self.history_lookup
        result.legal_actions_list = self.legal_actions_list
        result.info_state_per_player = self.info_state_per_player
        result.action_probabilities_table = self.action_probabilities_table
        result.game = self.game
        result.player_ids = self.player_ids
        return result



