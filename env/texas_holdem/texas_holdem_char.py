import env.environment as e
import numpy as np

# Phase
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
INT2STRING_PHASE = {PREFLOP: 'preflop', FLOP: 'flop',
                    TURN: 'turn', RIVER: 'river'}

# Player
CHANCE = -1
TERMINAL = -2
INT2STRING_PLAYER = {CHANCE: 'chance', TERMINAL: 'terminal', 0: 'p1', 1: 'p2'}

# Status
FOLDED = -2
NONRESPONSE = -1
CALLED = 0
RAISED1TIME = 1
RAISED2TIMES = 2
INT2STRING_STATUS = {FOLDED: 'folded', NONRESPONSE: 'non-response', CALLED: 'called',
                     RAISED1TIME: 'raised 1 time', RAISED2TIMES: 'raised 2 times'}

# Hand
INT2STRING_PATTERN = {0: 'C', 1: 'D', 2: 'H', 3: 'S'}
INT2STRING_FIGURE = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                     8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}
INT2STRING_CARD = {**{i: INT2STRING_PATTERN[i // 13] + INT2STRING_FIGURE[i % 13 + 1]
                      for i in range(52)}, -1: 'Null'}

# Action
FOLD = 0
CALL = 1
RAISE = 2
INT2STRING_ACTION = {FOLD: 'fold', CALL: 'call', RAISE: 'raise'}

# Card type
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIRS = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


class WorldState(e.WorldState):
    """World state object of env: Texas Hold'em."""

    def __init__(self, hand, pot, pub, phase, status, player):
        """Init the world state instance."""

        self.hand = hand
        self.pot = pot
        self.pub = pub
        self.phase = phase
        self.status = status
        self.player = player

    def legal_actions(self):
        """Return a list of actions that are legal on this state."""

        if self.player == TERMINAL:  # is terminal
            return []
        elif self.player == CHANCE:  # is chance
            if self.phase == PREFLOP:  # preflop
                return [Action(deal=[i, j, k, l])
                        for i in range(52) for j in range(52)
                        for k in range(52) for l in range(52)
                        if len(set([i, j, k, l])) == 4]
            elif self.phase == FLOP:  # flop
                return [Action(deal=[i, j, k])
                        for i in range(52) for j in range(52)
                        for k in range(52) if len(set([i, j, k])) == 3]
            elif self.phase == TURN:  # turn
                return [Action(deal=[i]) for i in set(range(52)) - set(self.pub)]
            else:  # river
                return [Action(deal=[i]) for i in set(range(52)) - set(self.pub)]
        else:
            if self.status[self.player] == RAISED2TIMES:  # can't raise
                return [Action(action=FOLD, player=self.player),
                        Action(action=CALL, player=self.player)]
            else:  # can raise
                if self.phase == PREFLOP or self.phase == FLOP:  # preflop or flop
                    return [Action(action=FOLD, player=self.player),
                            Action(action=CALL, player=self.player),
                            Action(action=RAISE, bet=1, player=self.player)]
                else:  # turn or river
                    return [Action(action=FOLD, player=self.player),
                            Action(action=CALL, player=self.player),
                            Action(action=RAISE, bet=2, player=self.player)]

    def chance_outcomes(self):
        """Return a list of actions and the corresponding probs."""

        assert self.is_chance()

        action_list = self.legal_actions()
        dealed_card = [*self.hand[0], *self.hand[1], *self.pub]  # dealed cards

        if self.phase == PREFLOP:  # preflop
            prob_list = [1 for _ in action_list]
        elif self.phase == FLOP:  # flop
            prob_list = [1 if all([d not in dealed_card for d in a.deal])
                         else 0 for a in action_list]
        elif self.phase == TURN:  # turn
            prob_list = [1 if all([d not in dealed_card for d in a.deal])
                         else 0 for a in action_list]
        else:  # river
            prob_list = [1 if all([d not in dealed_card for d in a.deal])
                         else 0 for a in action_list]
        prob_list = np.array(prob_list)
        prob_list = prob_list / np.sum(prob_list)

        return action_list, prob_list

    @property
    def winner(self):
        """Return the winner for a terminal world state."""

        assert self.is_terminal()

        # The player who fold will lose
        if self.status[0] == FOLDED:
            return 1
        if self.status[1] == FOLDED:
            return 0

        # Campare the hands
        card_type_0 = CardType([*self.hand[0], *self.pub])
        card_type_1 = CardType([*self.hand[1], *self.pub])

        def campare(a, b, *keys):
            if keys[0](a) > keys[0](b):
                return 0
            elif keys[0](a) < keys[0](b):
                return 1
            else:
                if len(keys) == 1:
                    return -1
                else:
                    return campare(a, b, *keys[1:])

        return campare(card_type_0, card_type_1, lambda x: x.type,
                       *[lambda x: x.key[i] for i in range(len(card_type_0.key))])

    def to_string(self):
        """Return a string representing this world state."""

        return '[[%s], [%s]], [%d, %d], [%s], %s, %s, [%s]' % \
            (', '.join(INT2STRING_CARD[h] for h in self.hand[0]),
             ', '.join(INT2STRING_CARD[h] for h in self.hand[1]),
             self.pot[0], self.pot[1],
             ', '.join(INT2STRING_CARD[p] for p in self.pub),
             INT2STRING_PHASE[self.phase],
             INT2STRING_PLAYER[self.player],
             ', '.join(INT2STRING_STATUS[s] for s in self.status))


class CardType(object):
    """Record the card type of a player."""

    def __init__(self, card_list):

        def pattern(x): return x // 13
        def figure(x): return (x + 12) % 13 + 2  # A = 14

        pattern_list = list(map(pattern, card_list))
        figure_list = list(map(figure, card_list))

        card_list.sort(reverse=True)
        pattern_list.sort(reverse=True)
        figure_list.sort(reverse=True)

        # Straight Flush
        for x in card_list[3:]:
            if x+1 in card_list and x+2 in card_list and x+3 in card_list:
                if x+4 in card_list and pattern(x) == pattern(x+4):  # A~9
                    self.type = STRAIGHT_FLUSH
                    self.key = [figure(x+4)]
                    return
                # C10, D10, H10, S10
                elif x-9 in card_list and x in [9, 22, 35, 48]:
                    self.type = STRAIGHT_FLUSH
                    self.key = [figure(x-9)]
                    return

        # Four of a Kind
        for x in figure_list:
            if figure_list.count(x) == 4:
                self.type = FOUR_OF_A_KIND
                self.key = [x, max([y for y in figure_list if y != x])]
                return

        # Full House
        for x in figure_list:
            if figure_list.count(x) == 3:
                for y in figure_list:
                    if y != x and figure_list.count(y) >= 2:
                        self.type = FULL_HOUSE
                        self.key = [x, y]
                        return

        # Flush
        for x in pattern_list:
            if pattern_list.count(x) >= 5:
                self.type = FLUSH
                same_list = [figure(y) for y in card_list if pattern(y) == x]
                same_list.sort(reverse=True)
                self.key = same_list[:5]
                return

        # Straight
        for x in figure_list:
            if x-1 in figure_list and x-2 in figure_list and x-3 in figure_list:
                if x-4 in figure_list or x+9 in figure_list:  # A: 14
                    self.type = STRAIGHT
                    self.key = x
                    return

        # Three of a kind
        for x in figure_list:
            if figure_list.count(x) == 3:
                self.type = THREE_OF_A_KIND
                self.key = [x] + [y for y in figure_list if y != x][:2]
                return

        # Two pairs
        for x in figure_list:
            if figure_list.count(x) == 2:
                for y in figure_list:
                    if y != x and figure_list.count(y) == 2:
                        self.type = TWO_PAIRS
                        self.key = [x, y, max([z for z in figure_list if z != x and z != y])]

        # One pair
        for x in figure_list:
            if figure_list.count(x) == 2:
                self.type = ONE_PAIR
                self.key = [x] + [y for y in figure_list if y != x][:3]

        # High Card
        self.type = HIGH_CARD
        self.key = figure_list[:5]
        return


class Action(e.Action):
    """Action object of env: Texas Hold'em."""

    def __init__(self, deal=None, action=None, bet=0, player=-1):
        """Init the action instance."""

        self.deal = deal  # list
        self.action = action  # scalar
        self.bet = bet
        self.player = player

    def to_string(self):
        """Return a string representing this action."""

        if self.player == CHANCE:  # is chance
            return ', '.join(INT2STRING_CARD[c] for c in self.deal)
        else:  # is not chance
            return INT2STRING_ACTION[self.action]


class PrivateObservation(e.Observation):
    """Private observation object of env: Texas Hold'em."""

    def __init__(self, hand, player):
        """Init the private observation instance."""

        self.hand = hand
        self.player = player

    def to_string(self):
        """Return a string representing this private observation."""

        return ', '.join(INT2STRING_CARD[h] for h in self.hand)


class PublicObservation(e.Observation):
    """Public observation object of env: Texas Hold'em."""

    def __init__(self, pot, pub):
        """Init the public observation instance."""

        self.pot = pot
        self.pub = pub

    def to_string(self):
        """Return a string representing this public observation."""

        return '[%s], [%s]' % \
            (', '.join(str(x) for x in self.pot),
             ', '.join(INT2STRING_CARD[p] for p in self.pub))
