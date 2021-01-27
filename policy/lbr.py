class LBRagent(object): # Only for Hold'em
    def __init__(self, game, idx, initial_infosate, opponent_policy):
        self.game = game
        self._idx = idx
        self._num_players = game.num_players
        self._infostate = initial_infosate
        self._histories = self._infostate.get_all_histories()
        self._opponent_range = [1 / len(self._histories) for k in self._histories] # uniform chance
        self._opponent_policy = opponent_policy
 
    def modify_range(self, action):
        current_player = self._histories[0].current_player()
        if current_player == self._idx or current_player == -1:
            self._histories = [h.child(action) for h in self._histories]

        else:
            for i in range(len(self._histories)):
                history = self._histories[i]
                p_a = self._opponent_policy.get_prob(history, action)
                self._opponent_range[i] = self._opponent_range[i] * p_a
                self._histories[i] = self._histories[i].child(action)
            self._opponent_range = self._opponent_range / sum(self._opponent_range)
    
    def step(self, history):
        self._infostate = history.get_info_state()
        histories = self._infostate.get_all_histories()
        assert histories == self._histories
        wp = 0
        for i in range(len(self._histories)):
            history = self._histories[i]
            while not history.is_terminal():
                if history.is_chance():
                    action = np.random.choice(history.chance_outcomes()[
                                          0], p=history.chance_outcomes()[1])
                    history = history.child(action)
                else:
                    action = history.legal_actions()[1] # Call
                    history = history.child(action)
            u = history.get_return()
            if (self._idx == 0 and u > 0) or (self._idx == 1 and u < 0):
                wp += self._opponent_range[i]
        legal_actions = self._infostate.legal_actions()
        util = {action.to_string(): 0 for action in legal_actions} # u(fold) = 0

        oppo_idx = 1 - self._idx
        asked = self._infostate.pot[oppo_idx] - self._infostate.pot[self._idx]
        util[legal_actions[1].to_string()] = wp * sum(self._infostate.pot) - (1 - wp) * asked

        if len(legal_actions) > 2:
            for action in legal_actions[2:]:
                fp = 0
                histories_after = []
                range_after = []
                for i in range(len(self._histories)):
                    p = self._opponent_range[i]
                    h = self._histories[i]
                    h_after = h.child(action)
                    p_fold = self._opponent_policy.get_prob(h_after, h_after.legal_actions()[0])

                    histories_after.append(h_after)
                    range_after.append(p * (1 - p_fold))

                    fp += p * p_fold
                
                range_after = range_after / sum(range_after)

                wp = 0
                for i in range(len(histories_after)):
                    history = histories_after[i]
                    while not history.is_terminal():
                        if history.is_chance():
                            action = np.random.choice(history.chance_outcomes()[
                                                0], p=history.chance_outcomes()[1])
                            history = history.child(action)
                        else:
                            action = history.legal_actions()[1] # Call
                            history = history.child(action)
                    u = history.get_return()
                    if (self._idx == 0 and u > 0) or (self._idx == 1 and u < 0):
                        wp += range_after[i]
                
                util[action.to_string()] = fp * sum(self._infostate.pot) + (1 - fp) * (wp * (sum(self._infostate.pot) + action.bet) - \
                                            (1 - wp) * (asked + action.bet) ) 
        if max(util.values()) == 0:
            return legal_actions[0] # fold


        value = 0
        for action in legal_actions:
            u = util[action.to_string()]
            if u > value:
                value = u
                selected_action = action
        return selected_action
            

  


                    